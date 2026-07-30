#!/usr/bin/env bash
# build.sh — 用 jetson-containers 組合出 DeepStream + PyTorch 容器。
# 在真 Jetson (JetPack 7.1) 上執行。詳見 handoff.md §5 / DEP-1。
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ds-torch-webcam}"

# jetson-containers 需在 PATH, 或以相對路徑呼叫。安裝方式:
#   git clone https://github.com/dusty-nv/jetson-containers
#   bash jetson-containers/install.sh
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
