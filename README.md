# Jetson DeepStream + PyTorch + Webcam（最小應用）

用 [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers) 在 **JetPack 7.1** 上跑的最小 demo：
`webcam → DeepStream/GStreamer → PyTorch 分類 → 輸出`。

**真正目的是測「套件依賴完不完整」**，不是做產品級 CV。完整決策過程與依賴問題追蹤在 [`handoff.md`](handoff.md)。

## 檔案
| 檔案 | 作用 |
|------|------|
| `handoff.md` | ⭐ 決策日誌 + 依賴問題追蹤 + 驗收標準（先看這個）|
| `app/preflight.py` | 依賴健檢 = 依賴完整性驗收（AC-1），永不 crash，最後給記分板 |
| `app/app.py` | 小應用：webcam→GStreamer/nv→appsink→PyTorch 分類（AC-2），headless 友善 |
| `build.sh` | `jetson-containers build deepstream pytorch torchvision` 組合容器 |
| `run.sh` | 在容器內跑 `preflight` / `app` / `shell`，自動 `autotag` + webcam 直通 |

## 快速開始（在 Jetson 上）
```bash
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
./build.sh
./run.sh preflight   # 先驗依賴
./run.sh app         # 再跑應用
```

## 已知最大風險
JetPack 7.1 = L4T r38 / CUDA 13 / Ubuntu 24.04；jetson-containers 的 DeepStream 打包對 JP7.1 的對應在寫作當下**尚未明確**（見 handoff `DEP-1`）。若組合建置失敗，走 handoff §7 的備援方案（NVIDIA 官方 DeepStream 8.0 JP7 容器當 base）。

## 可調環境變數
`VIDEO_DEVICE`（預設 `/dev/video0`）、`INFER_EVERY`（預設 15）、`RUN_SECONDS`（預設 60）、`CONTAINER_NAME`（預設 `ds-torch-webcam`）。
