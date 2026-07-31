# Jetson DeepStream + PyTorch + Webcam（最小應用）

在 **JetPack 6.2** 上，用 [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers) 跑一個
`webcam → DeepStream/GStreamer → PyTorch 分類 → 輸出` 的最小 demo。

**真正目的是測「套件依賴完不完整」**，不是做產品級 CV。完整決策過程與依賴問題追蹤在 [`handoff.md`](handoff.md)。

- 目標平台：**JetPack 6.2**（L4T r36.4.3 · CUDA 12.6 · Ubuntu 22.04 · Python 3.10）
- jetson-containers 對 JP6.2 選用：**DeepStream 8.0.0 / pyds 1.2.2**（`config.py` 的精確對應，非外推）

---

## 快速開始（在 JetPack 6.2 的 Jetson 上）

```bash
# 1) clone 本專案
git clone <your-repo-url> jetson-testing
cd jetson-testing

# 2) 安裝 jetson-containers 工具本身（不含 Docker；Docker 由 build.sh 檢查）
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh

# 3) 建置組合容器（DeepStream + PyTorch + torchvision）
./build.sh

# 4) 先驗依賴（= 依賴完整性驗收），再跑應用
./run.sh preflight
./run.sh app
```

> `build.sh` 步驟 0 會**冪等**檢查 Docker：JetPack 6.2 通常已內建就直接跳過，只有精簡映像缺 Docker 才會裝（需 sudo）。

---

## 檔案

| 檔案 | 作用 |
|------|------|
| [`handoff.md`](handoff.md) | ⭐ 決策日誌 + 依賴問題追蹤 + 驗收標準（先看這個）|
| [`build.sh`](build.sh) | 檢查 Docker → `jetson-containers build deepstream pytorch torchvision` 組合容器 |
| [`run.sh`](run.sh) | 在容器內跑 `preflight` / `app` / `shell`，自動 `autotag` + webcam 直通 |
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
