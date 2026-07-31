#!/usr/bin/env bash
# setup-host.sh — 一次性: 在 JP6.2 host 上裝好 Docker(官方, 含 buildx) + NVIDIA container runtime。
# 只需跑「一次」(裝好就永久)。之後用 ./build.sh 做 image、./run.sh 跑。
#
# 為何是官方 docker-ce 而非 Ubuntu 的 docker.io:
#   docker.io 不帶 buildx, 會讓後續 build 出現 exit 125 (見 handoff DEP-10)。
# 全程 set -e: 任一步真失敗就停, 不降級。
set -euo pipefail

echo "[setup] Docker(官方, 含 buildx) + NVIDIA runtime  (需 sudo 密碼)"

# 1. 基本工具 (base 映像常缺 curl)
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg git

# 2. 移除 Ubuntu 的 docker.io 及舊套件 (清舊版, 非 fallback; 沒裝也無妨)
sudo apt-get remove -y docker.io docker-doc docker-compose podman-docker containerd runc || true

# 3. 加 Docker 官方 apt 源 (--yes 讓重跑可覆寫金鑰)
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-jammy}")"
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update

# 4. 裝 docker-ce 全套 (engine + CLI + containerd + buildx + compose)
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker

# 5. 把「真正的你」加進 docker 群組 (sudo 下 $USER=root, 用 SUDO_USER)
TARGET_USER="${SUDO_USER:-$USER}"
sudo usermod -aG docker "${TARGET_USER}"

# 6. NVIDIA container runtime + 設成 default-runtime (容器要用 GPU)
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker

# 7. 驗證
echo "[setup] 驗證:"
docker --version
docker buildx version
echo "[setup]   default-runtime = $(docker info --format '{{.DefaultRuntime}}')"
echo
echo "[setup] 完成。接著:"
echo "        newgrp docker      # 讓 docker 群組生效 (不用重登入)"
echo "        ./build.sh         # 做 image (一次)"
echo "        ./run.sh preflight # 驗依賴"
