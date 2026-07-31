#!/usr/bin/env bash
# run.sh — 直接用 docker run 跑組合好的 image (preflight / app / shell)。
# 用法: ./run.sh [preflight|app|shell]
#
# 為何用 docker run 而非 jetson-containers run + autotag:
#   我們的 image 名稱固定 (build.sh --name=ds-torch-webcam), 且 autotag/jetson-containers run
#   之前在 sudo 汙染下踩到權限問題 + 會嘗試重建。改為明確 docker run, 可預期、不再繞路。
#   !! 不要用 sudo 跑 !! —— sudo 會把 jetson-containers/ 目錄弄成 root 擁有 (見 handoff T11)。
#   先確保在 docker 群組: 裝完 docker 後執行過 `newgrp docker` 或重新登入。
set -euo pipefail

MODE="${1:-preflight}"
CONTAINER_NAME="${CONTAINER_NAME:-ds-torch-webcam}"
VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# build.sh --name 產出的實際 tag 形如 ds-torch-webcam:r36.5.x-...; 抓第一個符合的。
IMAGE="$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep "^${CONTAINER_NAME}:" | head -1 || true)"
[ -z "${IMAGE}" ] && IMAGE="${CONTAINER_NAME}"

RUN_ARGS=(
  --runtime nvidia
  --network host
  -it --rm
  -v "${HERE}/app:/app"
  -e "VIDEO_DEVICE=${VIDEO_DEVICE}"
  -e "INFER_EVERY=${INFER_EVERY:-15}"
  -e "RUN_SECONDS=${RUN_SECONDS:-60}"
)

# webcam 直通 (host 上存在才加, 否則 docker run 會直接報錯)
if [ -e "${VIDEO_DEVICE}" ]; then
  RUN_ARGS+=(--device "${VIDEO_DEVICE}")
else
  echo "[run][WARN] ${VIDEO_DEVICE} 不存在, 不直通 webcam (preflight 的 webcam 檢查會 FAIL)。" >&2
fi

# X11 (有 DISPLAY 才加; headless 時 app 會自動只印文字)
if [ -n "${DISPLAY:-}" ]; then
  RUN_ARGS+=(-e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix)
fi

case "${MODE}" in
  preflight) CMD=(python3 /app/preflight.py) ;;
  app)       CMD=(python3 /app/app.py) ;;
  shell)     CMD=(bash) ;;
  *) echo "用法: $0 [preflight|app|shell]" >&2; exit 2 ;;
esac

echo "[run] image=${IMAGE} mode=${MODE} device=${VIDEO_DEVICE}"
exec docker run "${RUN_ARGS[@]}" "${IMAGE}" "${CMD[@]}"
