#!/usr/bin/env bash
# setup_nodo.sh -- prepara Ubuntu-22.04 (WSL2) como NODO (worker) del cluster.
# Lo ejecuta unirse_cluster.ps1 como root:
#   bash setup_nodo.sh <nombre> <archivo_clave_publica_del_maestro>
# No instala CUDA Toolkit: el binario que envia el maestro trae el runtime CUDA
# enlazado estaticamente; solo hace falta el driver NVIDIA de Windows.
set -e
NOMBRE="${1:?uso: setup_nodo.sh <nombre> <clave_maestro.pub>}"
CLAVE_FILE="${2:?falta el archivo con la clave publica del maestro}"
HOST="nodo-$NOMBRE"
MASTER_PUBKEY="$(tr -d '\r' < "$CLAVE_FILE" | grep -E '^(ssh-|ecdsa-)' | head -1)"
[ -n "$MASTER_PUBKEY" ] || { echo "ERROR: $CLAVE_FILE no contiene una clave SSH publica valida."; exit 1; }

echo "[1/6] GPU visible en WSL"
/usr/lib/wsl/lib/nvidia-smi -L || { echo "ERROR: WSL no ve la GPU. Actualiza el driver NVIDIA en Windows."; exit 1; }

echo "[2/6] Paquetes (OpenMPI 4.1.2, SSH, rsync)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq openmpi-bin libopenmpi-dev openssh-server rsync >/dev/null
mpirun --allow-run-as-root --version | head -1

echo "[3/6] Usuario mpiuser + clave del maestro"
id mpiuser >/dev/null 2>&1 || useradd -m -s /bin/bash mpiuser
install -d -m 700 -o mpiuser -g mpiuser /home/mpiuser/.ssh /home/mpiuser/.openmpi
install -d -o mpiuser -g mpiuser /home/mpiuser/llm_cluster
grep -qF "$MASTER_PUBKEY" /home/mpiuser/.ssh/authorized_keys 2>/dev/null || echo "$MASTER_PUBKEY" >> /home/mpiuser/.ssh/authorized_keys
chown mpiuser:mpiuser /home/mpiuser/.ssh/authorized_keys; chmod 600 /home/mpiuser/.ssh/authorized_keys

echo "[4/6] OpenMPI: solo interfaz Tailscale (100.64.0.0/10) y puertos fijos"
cat > /home/mpiuser/.openmpi/mca-params.conf <<EOF
btl_tcp_if_include = 100.64.0.0/10
oob_tcp_if_include = 100.64.0.0/10
btl_tcp_port_min_v4 = 10000
btl_tcp_port_range_v4 = 200
oob_tcp_dynamic_ipv4_ports = 10300-10400
btl_vader_single_copy_mechanism = none
plm_rsh_no_tree_spawn = 1
EOF
chown mpiuser:mpiuser /home/mpiuser/.openmpi/mca-params.conf
# hwloc (usado por orted) intenta abrir la pantalla X11 de WSLg y con red mirrored se cuelga: desactivar su plugin GL
grep -q HWLOC_COMPONENTS /etc/environment || echo 'HWLOC_COMPONENTS=-gl' >> /etc/environment

echo "[5/6] SSH (solo con clave, puerto 22)"
cat > /etc/ssh/sshd_config.d/10-cluster.conf <<EOF
Port 22
PubkeyAuthentication yes
PasswordAuthentication no
PermitRootLogin no
EOF

echo "[6/6] systemd + hostname $HOST"
touch /etc/wsl.conf
python3 - "$HOST" <<'PY'
import configparser, sys
c = configparser.ConfigParser(); c.optionxform = str
c.read("/etc/wsl.conf")
for s in ("boot", "network"):
    if not c.has_section(s): c.add_section(s)
c.set("boot", "systemd", "true")
c.set("network", "hostname", sys.argv[1])
with open("/etc/wsl.conf", "w") as f: c.write(f)
PY
grep -q "$HOST" /etc/hosts || echo "127.0.1.1 $HOST" >> /etc/hosts
systemctl enable ssh >/dev/null 2>&1 || true
echo "OK: nodo $HOST preparado (se aplica al reiniciar WSL)."
