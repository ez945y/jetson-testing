#!/usr/bin/env bash
# build.sh — 用 jetson-containers 組合出 DeepStream + PyTorch 容器。
# 目標: JetPack 6.2 (L4T r36.4.3 / CUDA 12.6 / Ubuntu 22.04)。詳見 handoff.md §5 / DEP-1。
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ds-torch-webcam}"

# ---------------------------------------------------------------------------
# 步驟 0: 安裝正規 Docker (官方源, 含 buildx) + NVIDIA container runtime
#
# 使用者要求「完整流程 / 整個重裝 / 不要 fallback」: 一律照 Docker 官方流程走一遍 ——
# 移除 Ubuntu 的 docker.io (不帶 buildx, 是 exit 125 的根因, 見 DEP-10), 改裝 docker-ce 全套。
# 全程 set -e: 任一步真失敗就停, 不降級、不繞路。跳過整段: SKIP_DOCKER_SETUP=1 ./build.sh
# ---------------------------------------------------------------------------
setup_step0() {
  echo "[build] 步驟 0: Docker(官方, 含 buildx) + NVIDIA runtime  (需 sudo)"

  # 0a. 基本工具 (base 映像常缺 curl)
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg git

  # 0b. 移除 Ubuntu 的 docker.io 及舊套件 (不帶 buildx; 見 DEP-10)。
  #     這行是「清掉舊版」, 非 fallback; 套件本來就沒裝也無妨, 故容忍非零。
  sudo apt-get remove -y docker.io docker-doc docker-compose podman-docker containerd runc || true

  # 0c. 加 Docker 官方 apt 源 (--yes 讓重跑時可覆寫金鑰, 支援「整個重裝」)
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-jammy}")"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${CODENAME} stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update

  # 0d. 裝 docker-ce 全套 (engine + CLI + containerd + buildx + compose)
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo systemctl enable --now docker

  # 0e. 把「真正的你」加進 docker 群組 (sudo 下 $USER=root, 用 SUDO_USER 抓真使用者)
  TARGET_USER="${SUDO_USER:-$USER}"
  sudo usermod -aG docker "${TARGET_USER}"

  # 0f. NVIDIA container runtime + 設成 default-runtime (jetson-containers build 階段就要 GPU)
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
  sudo systemctl restart docker

  # 0g. 驗證 (任一失敗即 set -e 中止, 不繼續往下 build)
  echo "[build] 驗證:"
  docker --version
  docker buildx version
  echo "[build]   default-runtime = $(docker info --format '{{.DefaultRuntime}}')"
  echo "[build] 步驟 0 完成。"
}

if [ "${SKIP_DOCKER_SETUP:-0}" != "1" ]; then
  setup_step0
fi

# ---------------------------------------------------------------------------
# 步驟 1: 用 jetson-containers 組合容器
# ---------------------------------------------------------------------------
# jetson-containers 需在 PATH, 或以相對路徑呼叫。安裝方式:
#   git clone https://github.com/dusty-nv/jetson-containers
#   bash jetson-containers/install.sh   (這步裝的是工具本身, 不含 Docker)
if ! command -v jetson-containers >/dev/null 2>&1; then
  echo "[build] 找不到 jetson-containers。請先安裝:" >&2
  echo "        git clone https://github.com/dusty-nv/jetson-containers" >&2
  echo "        bash jetson-containers/install.sh" >&2
  exit 1
fi

echo "[build] 更新 jetson-containers (DeepStream/JP7 打包更新頻繁, 見 DEP-1)..."
echo "[build] 若這是 git checkout, 建議先 'git -C <repo> pull'。"

echo "[build] 組合容器: deepstream + pytorch + torchvision -> ${CONTAINER_NAME}"
# 決策 D3/T1: 不寫死 image tag; autotag/build 依實機 L4T 自動挑基底。
jetson-containers build --name="${CONTAINER_NAME}" \
  deepstream \
  pytorch \
  torchvision

echo "[build] 完成。用 ./run.sh preflight 驗依賴。"
