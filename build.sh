#!/usr/bin/env bash
# build.sh — 用 jetson-containers 組合出 DeepStream + PyTorch 容器。
# 目標: JetPack 6.2 (L4T r36.4.3 / CUDA 12.6 / Ubuntu 22.04)。詳見 handoff.md §5 / DEP-1。
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ds-torch-webcam}"

# ---------------------------------------------------------------------------
# 步驟 0: 確認容器執行環境 (Docker + NVIDIA runtime)
#
# 目標平台 JP6.2 = Ubuntu 22.04。**JP6.2 base 映像常常連 curl 都沒有**,
# 所以這裡一律走 apt (base 一定有 apt), 不依賴 curl / get.docker.com。見 handoff DEP-8/DEP-9。
# 設計為 idempotent: 有就跳過, 不覆蓋 JetPack 既有的。跳過整段: SKIP_DOCKER_SETUP=1 ./build.sh
# ---------------------------------------------------------------------------
APT_UPDATED=0
apt_update_once() {  # 只 apt-get update 一次, 避免重複
  [ "${APT_UPDATED}" = "1" ] && return
  sudo apt-get update
  APT_UPDATED=1
}

ensure_base_tools() {
  # base 映像常缺 curl (使用者已回報)。git 通常有; 兩者缺才補。
  local miss=()
  command -v curl >/dev/null 2>&1 || miss+=(curl)
  command -v git  >/dev/null 2>&1 || miss+=(git)
  if [ "${#miss[@]}" -gt 0 ]; then
    echo "[build] 補基本工具 (base 映像缺): ${miss[*]}  (需 sudo)"
    apt_update_once
    sudo apt-get install -y "${miss[@]}"
  fi
}

ensure_docker() {
  if command -v docker >/dev/null 2>&1; then
    echo "[build] Docker 已存在: $(docker --version)"
    return
  fi
  echo "[build] 找不到 Docker。以 apt 安裝 docker.io (base 無 curl, 不用 get.docker.com)。"
  echo "[build] ↳ 需要 sudo 密碼; 這步會改動系統。"
  apt_update_once
  sudo apt-get install -y docker.io
  sudo systemctl enable --now docker || true
  sudo usermod -aG docker "$USER" || true
  echo "[build] Docker 安裝完成: $(docker --version 2>/dev/null || echo '需重登入生效')"
  echo "[build] 若稍後 docker 出現 permission denied, 請登出再登入 (docker 群組生效)。"
}

ensure_nvidia_runtime() {
  # 光有 docker 還不夠: run.sh 用 --runtime nvidia, 需要 NVIDIA container runtime。
  if docker info 2>/dev/null | grep -qiE 'runtimes:.*nvidia|nvidia'; then
    echo "[build] NVIDIA container runtime 已就緒。"
    return
  fi
  # 決策更新 (T9): 已鎖定 JP6.2 + 本機是 JetPack 裝置(L4T apt repo 已預配),
  # 此時正確套件明確就是 nvidia-container-toolkit, 不再有「裝到 x86 版」的歧義,
  # 故從「只警告」改為「apt 嘗試安裝, 失敗才警告」。
  echo "[build] 未偵測到 NVIDIA runtime。JP6.2 以 apt 安裝 nvidia-container-toolkit (L4T repo)。"
  echo "[build] ↳ 需要 sudo 密碼。"
  apt_update_once
  if sudo apt-get install -y nvidia-container-toolkit; then
    sudo nvidia-ctk runtime configure --runtime=docker || true
    sudo systemctl restart docker || true
    echo "[build] nvidia runtime 設定完成。"
  else
    echo "[build][WARN] apt 裝不到 nvidia-container-toolkit — L4T apt 來源可能沒配好。" >&2
    echo "[build][WARN] 確認 /etc/apt/sources.list.d/ 有 nvidia L4T repo 後重跑; 見 handoff DEP-8。" >&2
  fi
}

if [ "${SKIP_DOCKER_SETUP:-0}" != "1" ]; then
  echo "[build] 步驟 0/1: 確認基本工具 + Docker + NVIDIA runtime"
  ensure_base_tools
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
