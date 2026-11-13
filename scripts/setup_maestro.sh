#!/usr/bin/env bash
# setup_maestro.sh -- prepara Ubuntu-22.04 (WSL2) como MAESTRO del cluster.
# Lo ejecuta unirse_cluster.ps1 -Rol maestro como root:
#   bash setup_maestro.sh <ruta_wsl_del_repo>
# Instala CUDA Toolkit 12.6 (repositorio WSL-Ubuntu de NVIDIA), OpenMPI, SSH,
# numpy/matplotlib; crea mpiuser con su clave SSH; compila el programa y genera
# las matrices en /home/mpiuser/llm_cluster; deja la clave publica en
# <repo>/scripts/clave_maestro.pub para entregarla a los nodos.
set -e
REPO="${1:?uso: setup_maestro.sh <ruta_wsl_del_repo>}"
CUDA_VER=12-6
export DEBIAN_FRONTEND=noninteractive

echo "[1/7] GPU visible en WSL"
/usr/lib/wsl/lib/nvidia-smi -L || { echo "ERROR: WSL no ve la GPU. Actualiza el driver NVIDIA en Windows."; exit 1; }

echo "[2/7] Paquetes (OpenMPI 4.1.2, SSH, rsync, numpy, matplotlib)"
apt-get update -qq
apt-get install -y -qq openmpi-bin libopenmpi-dev openssh-server rsync wget \
  python3-numpy python3-matplotlib >/dev/null

echo "[3/7] CUDA Toolkit ($CUDA_VER, repositorio WSL-Ubuntu; NO instala driver Linux)"
if [ ! -x /usr/local/cuda/bin/nvcc ]; then
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
  apt-get update -qq
  apt-get install -y -qq cuda-toolkit-$CUDA_VER >/dev/null
fi
/usr/local/cuda/bin/nvcc --version | tail -1

echo "[4/7] Usuario mpiuser, clave SSH y configuracion OpenMPI"
id mpiuser >/dev/null 2>&1 || useradd -m -s /bin/bash mpiuser
install -d -m 700 -o mpiuser -g mpiuser /home/mpiuser/.ssh /home/mpiuser/.openmpi
[ -f /home/mpiuser/.ssh/id_ed25519 ] || sudo -u mpiuser ssh-keygen -q -t ed25519 -N '' -f /home/mpiuser/.ssh/id_ed25519 -C mpiuser@master
cat > /home/mpiuser/.ssh/config <<EOF
Host 100.* nodo-* master
    User mpiuser
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    ConnectTimeout 8
EOF
cat > /home/mpiuser/.openmpi/mca-params.conf <<EOF
btl_tcp_if_include = 100.64.0.0/10
oob_tcp_if_include = 100.64.0.0/10
btl_tcp_port_min_v4 = 10000
btl_tcp_port_range_v4 = 200
oob_tcp_dynamic_ipv4_ports = 10300-10400
btl_vader_single_copy_mechanism = none
plm_rsh_no_tree_spawn = 1
EOF
grep -q 'cuda/bin' /home/mpiuser/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /home/mpiuser/.bashrc
chown -R mpiuser:mpiuser /home/mpiuser/.ssh /home/mpiuser/.openmpi; chmod 600 /home/mpiuser/.ssh/config

echo "[5/7] SSH + systemd + hostname master"
cat > /etc/ssh/sshd_config.d/10-cluster.conf <<EOF
Port 22
PubkeyAuthentication yes
PasswordAuthentication no
PermitRootLogin no
EOF
touch /etc/wsl.conf
python3 - <<'PY'
import configparser
c = configparser.ConfigParser(); c.optionxform = str
c.read("/etc/wsl.conf")
for s in ("boot", "network"):
    if not c.has_section(s): c.add_section(s)
c.set("boot", "systemd", "true")
c.set("network", "hostname", "master")
with open("/etc/wsl.conf", "w") as f: c.write(f)
PY
grep -q ' master$' /etc/hosts || echo "127.0.1.1 master" >> /etc/hosts
systemctl enable ssh >/dev/null 2>&1 || true

echo "[6/7] Compilar programa y generar matrices en ~/llm_cluster"
sudo -u mpiuser bash -s "$REPO" <<'EOS'
set -e
REPO="$1"; W=/home/mpiuser/llm_cluster
mkdir -p "$W"; cd "$W"
cp "$REPO"/codigo/* "$REPO"/scripts/cluster.sh .
sed -i 's/\r$//' *.sh *.py
chmod +x *.sh
export PATH=/usr/local/cuda/bin:$PATH
GEN="-gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_89,code=compute_89"
nvcc -O3 $GEN -ccbin=mpicxx -cudart static llm_mpi_cuda_gpt2small.cu -o llm_mpi_cuda_gpt2small
mpicxx -O2 validate_cpu_mpi_gpt2small.cpp -o validate_cpu_mpi_gpt2small
[ -f matrices/Wvocab.bin ] || python3 generate_matrices.py --out-dir matrices --n-seq 128 --seed 42 | tail -1
mpirun -np 1 ./llm_mpi_cuda_gpt2small --matrices-dir matrices --repeats 2 --results-csv /tmp/prueba.csv | grep -E 'Rank|ganador'
EOS

echo "[7/7] Clave publica del maestro"
cp /home/mpiuser/.ssh/id_ed25519.pub "$REPO/scripts/clave_maestro.pub"
cat /home/mpiuser/.ssh/id_ed25519.pub
echo "OK: maestro preparado. Clave guardada en scripts/clave_maestro.pub"
