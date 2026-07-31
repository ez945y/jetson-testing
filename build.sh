#!/usr/bin/env bash
# build.sh — 用 jetson-containers 組合出 DeepStream + PyTorch 容器。
# 目標: JetPack 6.2 (L4T r36.4.3 / CUDA 12.6 / Ubuntu 22.04)。詳見 handoff.md §5 / DEP-1。
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ds-torch-webcam}"

# ---------------------------------------------------------------------------
# 步驟 0: 確認容器執行環境 (Docker + NVIDIA runtime)
#
# 註: 正常燒錄的 JetPack 7.1 「已預裝」Docker 與 nvidia runtime, 此段通常整段跳過。
#     只有精簡 / 自訂 / 被移除的映像才會真的去裝。設計為 idempotent: 有就跳過, 不覆蓋。
#     跳過此段: SKIP_DOCKER_SETUP=1 ./build.sh
# ---------------------------------------------------------------------------
ensure_docker() {
  if command -v docker >/dev/null 2>&1; then
    echo "[build] Docker 已存在: $(docker --version)"
    return
  fi
  echo "[build] 找不到 Docker (JetPack 通常預裝, 此機可能是精簡映像)。開始安裝..."
  echo "[build] ↳ 需要 sudo 密碼; 這步會改動系統。"
  # Ubuntu 24.04 / arm64: 用 Docker 官方 convenience script (支援 Jetson arm64)。
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sudo sh /tmp/get-docker.sh
  # 讓目前使用者免 sudo 用 docker (需登出再登入才生效)。
  sudo usermod -aG docker "$USER" || true
  echo "[build] Docker 安裝完成。若稍後 docker 出現 permission denied, 請登出再登入。"
}

ensure_nvidia_runtime() {
  # 光有 docker 還不夠: run.sh 用 --runtime nvidia, 需要 NVIDIA container runtime。
  if docker info 2>/dev/null | grep -qiE 'runtimes:.*nvidia|nvidia'; then
    echo "[build] NVIDIA container runtime 已就緒。"
    return
  fi
  # 決策: 這裡「只警告不強裝」。Jetson 的 nvidia runtime 應由 JetPack 提供
  #       (apt 套件 nvidia-container / L4T repo), 與 x86 的 nvidia-container-toolkit
  #       來源不同; 由腳本亂裝可能裝錯版本、弄壞 JetPack。故留給人工確認。
  echo "[build][WARN] 未偵測到 NVIDIA container runtime。GPU / DeepStream 會無法運作。" >&2
  echo "[build][WARN] 正常 JetPack 應已內建。若確實缺, 用 JetPack 的來源補裝, 例如:" >&2
  echo "              sudo apt-get update && sudo apt-get install -y nvidia-container" >&2
  echo "              sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker" >&2
  echo "[build][WARN] (刻意不自動安裝, 避免在 Jetson 上裝到 x86 版本; 見 handoff DEP-8)" >&2
}

if [ "${SKIP_DOCKER_SETUP:-0}" != "1" ]; then
  echo "[build] 步驟 0/1: 確認 Docker + NVIDIA runtime"
  ensure_docker
  ensure_nvidia_runtime
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
