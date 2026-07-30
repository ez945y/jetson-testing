#!/usr/bin/env bash
# run.sh — 在 jetson-containers 容器內跑 preflight / app / shell。
# 用法: ./run.sh [preflight|app|shell]
set -euo pipefail

MODE="${1:-preflight}"
CONTAINER_NAME="${CONTAINER_NAME:-ds-torch-webcam}"
VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v jetson-containers >/dev/null 2>&1; then
  echo "[run] 找不到 jetson-containers。先跑 ./build.sh 的前置安裝步驟。" >&2
  exit 1
fi

# 決策 D3: 用 autotag 依實機 L4T 挑對的 image; 允許 CONTAINER_NAME 覆寫。
IMAGE="$(autotag "${CONTAINER_NAME}" 2>/dev/null || echo "${CONTAINER_NAME}")"
echo "[run] image = ${IMAGE}"

# DEP-6/DEP-7: 顯式直通 webcam 與 X11 (headless 也能跑, app 會自動判斷 DISPLAY)。
EXTRA_ARGS=(
  --device "${VIDEO_DEVICE}:${VIDEO_DEVICE}"
  -e "VIDEO_DEVICE=${VIDEO_DEVICE}"
  -e "INFER_EVERY=${INFER_EVERY:-15}"
  -e "RUN_SECONDS=${RUN_SECONDS:-60}"
  -v "${HERE}/app:/workspace/app"
)
if [ -n "${DISPLAY:-}" ]; then
  EXTRA_ARGS+=(-e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix)
fi

case "${MODE}" in
  preflight) CMD="python3 /workspace/app/preflight.py" ;;
  app)       CMD="python3 /workspace/app/app.py" ;;
  shell)     CMD="bash" ;;
  *) echo "用法: $0 [preflight|app|shell]" >&2; exit 2 ;;
esac

echo "[run] mode=${MODE} device=${VIDEO_DEVICE}"
exec jetson-containers run "${EXTRA_ARGS[@]}" "${IMAGE}" ${CMD}
