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
# 步驟 1: 用 jetson-containers「融合」—— 拿現成 deepstream 當 --base, 只疊 pytorch
#
# 為何用 --base (見 handoff DEP-11 / T11):
#   你的 L4T r36.5 (JP6.2.1) 沒有預建 deepstream。直接
#   `jetson-containers build deepstream pytorch` 會從 ubuntu:22.04 把整條 19 層「全從源碼編」(數小時)。
#   改成把「已預建、實測可跑」的 deepstream image 當地基, jetson-containers 只需疊 pytorch 那一層
#   (裝預建 wheel, 幾分鐘)。這才是 jetson-containers 該有的「快速融合」用法。
#
# 基底二選一 (用環境變數切換):
#   預設 = dustynv/deepstream:r36.2.0  (DeepStream 6.4.0 / CUDA 12.2; 已實測 pyds+gstreamer 全過, 免 NGC 登入)
#   最新 = NVIDIA 官方, 與 host cu126 原生對齊 (需先 `docker login nvcr.io`, NGC 免費帳號):
#       DEEPSTREAM_BASE=nvcr.io/nvidia/deepstream:7.1-triton-multiarch CUDA_VERSION=12.6 ./build.sh
# ---------------------------------------------------------------------------
DEEPSTREAM_BASE="${DEEPSTREAM_BASE:-dustynv/deepstream:r36.2.0}"
CUDA_VERSION="${CUDA_VERSION:-12.2}"   # 必須對齊基底容器的 CUDA (r36.2.0=12.2); 用 NVIDIA 基底請設 12.6

if ! command -v jetson-containers >/dev/null 2>&1; then
  echo "[build] 找不到 jetson-containers。請先安裝:" >&2
  echo "        git clone https://github.com/dusty-nv/jetson-containers" >&2
  echo "        bash jetson-containers/install.sh" >&2
  exit 1
fi

echo "[build] 融合: pytorch + torchvision 疊到 ${DEEPSTREAM_BASE} (CUDA ${CUDA_VERSION}) -> ${CONTAINER_NAME}"
CUDA_VERSION="${CUDA_VERSION}" jetson-containers build \
  --base="${DEEPSTREAM_BASE}" \
  --name="${CONTAINER_NAME}" \
  pytorch torchvision

echo "[build] 完成。用 ./run.sh preflight 驗依賴。"
