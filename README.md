# Jetson DeepStream + PyTorch + Webcam（最小應用）

在 **JetPack 6.2** 上跑一個 `webcam → DeepStream/GStreamer → PyTorch 分類 → 輸出` 的最小 demo。

**真正目的是測「套件依賴完不完整」**，不是做產品級 CV。完整決策過程與依賴問題追蹤在 [`handoff.md`](handoff.md)。

- 目標平台：**JetPack 6.2.x**（實機 L4T r36.5 · CUDA 12.6 · Ubuntu 22.04 · Python 3.10）
- 實際堆疊：**官方 DeepStream 7.1 容器** + **torch 2.11 / torchvision 0.26**（jetson-ai-lab cu126 預建 wheel）+ **pyds 1.2.0**（官方 wheel）
- 補的系統庫（官方 DS 容器缺）：`libopenblas0`、`cudss`、`libcusparselt0`、`python3-gi`（詳見 handoff DEP-13/14）
- （最初嘗試 `jetson-containers build` 全從源碼編譯的路線已棄用，原因見 handoff DEP-11/T12）

---

## 快速開始（在 JetPack 6.2 的 Jetson 上）

```bash
# 1) clone 本專案
git clone <your-repo-url> jetson-testing
cd jetson-testing

# 2) 一次性: 裝 Docker(官方, 含 buildx) + NVIDIA runtime, 然後讓 docker 群組生效
./setup-host.sh
newgrp docker

# 3) 登入 NGC (拉 NVIDIA 官方 DeepStream 容器用; 免費帳號, API key 到 ngc.nvidia.com 申請)
docker login nvcr.io    # username 固定填 $oauthtoken, password 填 NGC API key

# 4) 做 image (一次): 官方 DeepStream 7.1 + pip 疊 jetson 預建 torch wheel, 零編譯
./build.sh

# 5) 先驗依賴（= 依賴完整性驗收），再跑應用
./run.sh preflight
./run.sh app
```

> **不要用 `sudo` 跑 `build.sh` / `run.sh`**：sudo 會把目錄弄成 root 擁有、之後非 sudo 執行全卡權限（見 handoff T11）。`setup-host.sh` 裝完後用 `newgrp docker` 讓群組生效即可。
>
> **這一顆 image 是什麼**：沒有任何官方現成 image 同時含 DeepStream + PyTorch，所以 `build.sh` 用薄 [`Dockerfile`](Dockerfile) 自己合一顆——`FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch`（**官方對 JP6.2 / L4T 36.4–36.5 的 release**，CUDA 12.6 與 host 對齊）+ `pip install torch torchvision`（jetson-ai-lab 的 **cu126 預建 wheel**）。全程零編譯，image 做一次永久重用。
>
> **不用 `jetson-containers build`**：實測它必從源碼重建依賴樹（含 CUDA）——r36.5 無預建時整條編數小時、且在 cuda 階段 `exit 100`（見 handoff DEP-11/T12）。
>
> **免登入備援**：dustynv 的 r36.2 預建（CUDA 12.2 較舊，已實測 pyds/gstreamer 在 r36.5 host 全過）：
> ```bash
> DEEPSTREAM_BASE=dustynv/deepstream:r36.2.0 TORCH_INDEX= ./build.sh
> ```

---

## 檔案

| 檔案 | 作用 |
|------|------|
| [`handoff.md`](handoff.md) | ⭐ 決策日誌 + 依賴問題追蹤 + 驗收標準（先看這個）|
| [`setup-host.sh`](setup-host.sh) | 一次性：host 裝官方 Docker（含 buildx）+ NVIDIA container runtime |
| [`Dockerfile`](Dockerfile) | 薄層融合：`FROM` 官方 DeepStream 7.1 + pip 疊預建 torch wheel |
| [`build.sh`](build.sh) | `docker build` 做出 `ds-torch-webcam` image（一次，永久重用）|
| [`run.sh`](run.sh) | `docker run` 跑 `preflight` / `app` / `shell`，webcam 直通 |
| [`app/preflight.py`](app/preflight.py) | 依賴健檢 = 依賴完整性驗收（AC-1），永不 crash，最後給記分板 |
| [`app/app.py`](app/app.py) | 小應用：webcam→GStreamer/nv→appsink→PyTorch 分類（AC-2），headless 友善 |

---

## 用法

```bash
./run.sh preflight   # 依賴健檢，先跑這個
./run.sh app         # 跑應用（預設讀 /dev/video0，跑 60 秒）
./run.sh shell       # 手動進容器
```

可調環境變數（覆寫即可）：

| 變數 | 預設 | 說明 |
|------|------|------|
| `VIDEO_DEVICE` | `/dev/video0` | 用哪支 webcam |
| `INFER_EVERY` | `15` | 每幾幀跑一次 PyTorch 分類 |
| `RUN_SECONDS` | `60` | 跑幾秒自動結束（0 = 不限）|
| `CONTAINER_NAME` | `ds-torch-webcam` | 組合容器名 |
| `SKIP_DOCKER_SETUP` | `0` | 設 `1` 跳過 build.sh 的 Docker 檢查 |

範例：
```bash
VIDEO_DEVICE=/dev/video2 RUN_SECONDS=0 ./run.sh app
```

---

## 遠端連線（SSH）

在沒有螢幕、或想從電腦操作 Jetson 時：

```bash
# 在 Jetson 上：查內網 IP
hostname -I

# 在 Jetson 上：確認 SSH server（base 映像可能沒裝）
sudo apt-get install -y openssh-server && sudo systemctl enable --now ssh
```

從你的電腦連（帳號 `mike`、主機名 `ubuntu`）：
```bash
ssh mike@<jetson-ip>
```
主機名是 `ubuntu` 且 mDNS 有開時，可不記 IP：
```bash
ssh mike@ubuntu.local
```
免密碼（選用，在你電腦上跑一次）：
```bash
ssh-copy-id mike@<jetson-ip>
```

---

## 疑難排解

### `build.sh` 失敗、結尾是 `returned non-zero exit status 125`
`exit 125` 是 **Docker 自己起不動那個 build step**，通常掛在最基礎的 `build-essential` 階段 —— 代表**還沒碰到 deepstream/pytorch，是 `docker buildx` 層就失敗**，跟套件依賴無關。詳見 `handoff.md` DEP-10。

先看**真正的錯誤訊息**（traceback 只是外層），在 jetson-containers 的 log 檔：
```bash
tail -n 30 jetson-containers/logs/*/build/*build-essential.txt
```

再檢查兩個最常見前提：
```bash
docker buildx version          # 報錯 → buildx 沒裝 (Ubuntu 的 docker.io 常缺)
```
```bash
cat /etc/docker/daemon.json    # 應含 "default-runtime": "nvidia"
```

**修法：`build.sh` 步驟 0 現在已用 Docker 官方源裝 `docker-ce` 全套（含 buildx）並設好 `default-runtime: nvidia`，正常重跑一次就好：**
```bash
sudo ./build.sh
```
> 步驟 0 會移除 Ubuntu 的 `docker.io`（buildx 缺失的根因，見 `handoff.md` DEP-10/T10），改裝官方 `docker-ce`。全程 `set -e`，任一步失敗即停、不降級。

---

## 為什麼選 JetPack 6.2（而非 7.1）

JP6.2 是 **L4T r36.4.3**，正好是 jetson-containers `deepstream/config.py` 明確對應 DeepStream 8.0.0 的那一筆——
屬於**已驗證的相容組合**。相對地 JP7.1（L4T r38 / CUDA 13）在該版本表**沒有專屬條目**，會被 `>=` 級聯
「樂觀地」對到同一筆 JP6.2 建置，存在靜默誤配風險（詳見 `handoff.md` DEP-1）。選 JP6.2 直接避開這個坑。
