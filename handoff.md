# Handoff — Jetson DeepStream + PyTorch + Webcam 小應用

> 目的：用 [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers) 在 **JetPack 7.1** 上，跑一個
> 「webcam → DeepStream (GStreamer) → PyTorch 分類 → 疊字輸出」的最小應用，
> **主要用來驗證套件依賴（DeepStream / pyds / PyTorch / GStreamer / v4l2）在 JP7.1 上到底完不完整。**
>
> 撰寫者立場：資深工程師。以下記錄「決策過程」與「決策如何隨資訊改變」，而不是只記結果。
> 環境：本文件在一台 **非 Jetson** 的 macOS 開發機上撰寫，無法實機執行；因此程式碼以
> **可攜、可自我診斷、可優雅降級（graceful degradation）** 為第一原則。

---

## 0. 一句話結論（給趕時間的人）

- 先跑 `app/preflight.py`（依賴健檢），**它就是「依賴完不完整」的驗收工具**。
- 再跑 `app/app.py`（真正的小應用）。
- 兩者都設計成：**缺哪個依賴就明確告訴你缺哪個**，而不是丟一坨 traceback。
- **最大風險**：JP7.1（L4T r38 / CUDA 13）對 DeepStream 8.0 的 jetson-containers 打包在寫作當下**尚未有明確對應條目**，可能需要手動處理。詳見 §3。

---

## 1. 需求拆解與驗收標準（自訂）

使用者要「簡單應用 + 測依賴完整性」，所以我把驗收標準訂成兩層：

### AC-1　依賴健檢（必須全過才算「依賴完整」）
| # | 檢查項 | 通過條件 |
|---|--------|----------|
| 1 | `import torch` | 成功，且 `torch.cuda.is_available() == True` |
| 2 | torch 看得到 GPU | `torch.cuda.get_device_name(0)` 回傳 Jetson GPU（Orin/Thor）|
| 3 | `import pyds` | 成功（DeepStream Python bindings 有裝）|
| 4 | GStreamer + nvidia 外掛 | `gst-inspect-1.0 nvinfer` / `nvvideoconvert` 存在 |
| 5 | Webcam 節點 | `/dev/video0`（或指定節點）存在且可開 |
| 6 | DeepStream 版本 | `deepstream-app --version` 可執行，版本 ≥ 7.1（JP7 應為 8.0）|

### AC-2　端到端小應用（Demo 能動）
| # | 檢查項 | 通過條件 |
|---|--------|----------|
| 1 | 開 webcam | GStreamer pipeline 從 v4l2 取得畫面不報錯 |
| 2 | DeepStream 參與 | 用 nv 元件（`nvvideoconvert` 等）處理畫面，`appsink` 取得 frame |
| 3 | PyTorch 推論 | 對每 N 幀跑一次分類（MobileNetV3/ResNet18），輸出 top-1 標籤 |
| 4 | 結果可見 | 終端持續印出 `frame N: <label> (p=..)`，或（有顯示器時）疊字視窗 |
| 5 | 穩定性 | 連續跑 ≥ 60 秒不崩、FPS 印在 log |

> 決策：把「DeepStream 做 nvinfer 硬體推論」與「PyTorch 做推論」**兩條路合併成一個 pipeline** 會很複雜、又難在無實機下驗證。
> 因此小應用的職責分工定為：**DeepStream/GStreamer 負責 webcam 擷取與 GPU 影像前處理 → appsink 交給 PyTorch 做分類**。
> 這樣三個依賴（DeepStream 元件、pyds/GStreamer、PyTorch）**都被實際載入並使用到**，達成「測依賴」目的，又保持簡單。

---

## 2. 架構決策（含被我否決的方案）

**選定架構：**
```
v4l2src (webcam) → nvvideoconvert → capsfilter(RGBA) → appsink
                                                          │  (numpy frame)
                                                          ▼
                                            PyTorch 分類 (MobileNetV3-Small)
                                                          │
                                                          ▼
                                          終端輸出 top-1 標籤 / (可選) 疊字
```

**決策紀錄：**

- **D1｜為何 appsink 而非 DeepStream 原生 nvinfer？**
  原生 nvinfer 需要 TensorRT engine + 配置檔 + label 檔，對「只是測依賴」而言過重，且我無實機可轉 engine。
  改用 `appsink → PyTorch`，用 torchvision 的 pretrained 權重（首次執行自動下載），零額外資產。
  → 代價：少驗證到 nvinfer 的 TensorRT 路徑。折衷：preflight 用 `gst-inspect-1.0 nvinfer` **靜態驗證** nvinfer 存在（AC-1 #4），不在主流程跑它。

- **D2｜webcam 來源用 `v4l2src` 還是 `nvv4l2camerasrc`？**
  `nvv4l2camerasrc` 是 Jetson CSI/argus 相機用；一般 USB webcam 走 `v4l2src`。
  → 決策：**預設 `v4l2src`（USB webcam 最通用），並在 app 內做元件可用性偵測**，找不到就換路徑。避免綁死某型相機。

- **D3｜為何不寫死容器 tag？**
  jetson-containers 的 `autotag` 會依實機 L4T 版本挑對的 image。寫死 tag 在跨 JetPack 版本必爆。
  → 決策：`run.sh` 一律用 `autotag`，並容許用環境變數覆寫。

- **D4｜PyTorch 模型選誰？**
  MobileNetV3-Small：小、下載快、CPU/GPU 都能跑、ImageNet 1000 類足夠 demo。
  ResNet18 作為備援（若 mobilenet 權重下載失敗）。

- **D5｜無實機如何「交付可用代碼」？**
  所有腳本以「**能在真 Jetson 上一鍵跑，且失敗訊息可讀**」為準。
  程式大量使用「偵測 → 若缺則明確報錯並給修復指令」的模式，讓實機那端的人不用回頭問我。

---

## 3. 套件依賴問題追蹤（本任務核心）

> 這節是使用者最在意的：**依賴完不完整**。以下為「規劃時預判 + 需實機確認」的清單。
> 狀態：🔴 已知風險 / 🟡 待實機確認 / 🟢 已確認可行（實機驗過再改）

| ID | 依賴問題 | 狀態 | 說明與應對 |
|----|----------|------|-----------|
| DEP-1 | **DeepStream 8.0 對 JP7.1 (L4T r38) 的 jetson-containers 打包 —— 樂觀誤配風險** | 🔴 | **機制已查證**（見 `config.py`）：選版是一串 `>=` 由高到低的 if-elif，最上面是 `L4T_VERSION >= 36.4.3 → DS 8.0.0 / pyds 1.2.2 / tarball 'deepstream_sdk_v8.0.0_jetson.tbz2'`（此 tarball 為 **JP6.2 / CUDA 12.6 / Ubuntu 22.04** 建）。JP7.1 是 L4T **r38**，`38 >= 36.4.3` 成立 → **命中最上面那個 JP6.2 分支**，不是落到 `else: package=None`。也就是說 **沒有版本上限保護**：比最新已知還新的 L4T 會被「樂觀地」對到最新已知的 JP6.2 建置。危險點是 **它不會報「不支援」，而是靜默裝一個 CUDA 12.6 的 DeepStream 到 CUDA 13 環境**，錯誤延後到 `install.sh` / pyds 編譯 / `import pyds` / runtime 才炸（看起來像成功）。`else→None` 只在版本 **< r32.6**（比最舊還舊）時才觸發。**應對**：(a) 實機先 `git -C jetson-containers pull`——此表更新頻繁，官方隨時可能補 r38 條目；(b) `preflight.py` 專門抓這種假成功（`import pyds` / `deepstream-app --version`）；(c) 若確認誤配，走 §7 備援：改用 NVIDIA NGC 官方 DeepStream 8.0 JP7 容器當 base。 |
| DEP-2 | **pyds（DeepStream Python bindings）** | 🟡 | JP7 對應 pyds 版本較新（DS8.0 → pyds ≥1.2.x）。jetson-containers 的 deepstream 包會在版本 ≥1.2.0 時**從源碼編譯** gst-python + bindings（需 meson/cmake/pybind11）。編譯鏈缺任一都會失敗。preflight 會直接 `import pyds` 抓這個。 |
| DEP-3 | **PyTorch on CUDA 13 / JP7** | 🟡 | jetson-containers 官方已宣稱支援 JetPack 7 (CUDA 13.x)。風險在 torchvision 版本需與 torch 對齊，否則 `import torchvision` 會因 C++ ABI 不合而炸。**應對**：app 對 torchvision 匯入做 try/except，失敗時降級為「不做分類、只驗證擷取路徑」。 |
| DEP-4 | **Python 版本落差** | 🟡 | JP7.1 基底 Ubuntu 24.04 → Python 3.12；pyds wheel/編譯需對上 3.12。若某包只出 3.10 wheel 會裝不上。preflight 印出 `sys.version` 方便比對。 |
| DEP-5 | **GStreamer nv 外掛註冊** | 🟡 | 容器內 `nvvideoconvert`/`nvstreammux`/`nvinfer` 需在 `GST_PLUGIN_PATH` 註冊。DeepStream install.sh 會處理，但若 base image 換過可能漏。preflight 用 `gst-inspect-1.0` 逐一檢查。 |
| DEP-6 | **webcam 裝置直通** | 🟡 | 容器要 `--device /dev/video0`。`jetson-containers run` 預設不一定帶。`run.sh` 已顯式加上 `--device` 與 X11 掛載。 |
| DEP-7 | **顯示 / X11** | 🟡 | 無螢幕或 headless 時 `nveglglessink`/imshow 會爆。app 預設 **headless 友善**：無 DISPLAY 就只印文字，不開視窗。 |

**追蹤方式**：每個 DEP 在實機驗過就把狀態改成 🟢 並補一行「實機結果」。這份表就是依賴完整性的最終記分板。

---

## 4. 決策如何「改變」（時間線 / changelog）

> 使用者明講：想看「我的決策發生什麼改變」。這裡逐步記錄想法轉折。

- **T0 初始假設**：以為 jetson-containers 直接有現成 `deepstream+pytorch` 合體 image，`autotag` 一下就能跑。
- **T1 修正**：查 repo 後確認 —— 沒有現成合體 image，需 `jetson-containers build --name=... deepstream pytorch` **自行組合**。決策：改寫 `build.sh` 做組合建置。
- **T2 重大修正（影響架構）**：查 `deepstream/config.py` 發現**版本表沒有 r38/JP7 專屬條目**（DEP-1）。
  → 決策從「假設依賴完整、直接跑」轉為「**先健檢、預期會缺、把缺料變成可讀報告**」。這就是 `preflight.py` 誕生的原因。
- **T3 架構收斂**：原本想做 DeepStream 原生 nvinfer 全硬體 pipeline；評估「無實機 + 只為測依賴」後**否決**，改 appsink+PyTorch（D1）。決策從「秀肌肉」轉為「最小可驗證」。
- **T4 韌性優先**：因無法實機除錯，決定所有依賴匯入都包 try/except 且**分層降級**，寧可少做功能也要能跑到「告訴你哪裡缺」。
- **T5 開發機煙霧測試（2026-07-30）**：在 macOS 開發機（Python 3.12，無任何 Jetson 依賴）跑 `preflight.py`：**不崩、跑到底、印出記分板、exit 1**，正確標示 torch/pyds/deepstream/gst/webcam 全缺。證明「缺料變可讀報告」的設計成立。`app.py` 通過 `py_compile`。實機只是把這些 FAIL 逐一翻成 PASS 的過程。
- **T6 選版機制查證（修正 DEP-1 的認知）**：原本 DEP-1 寫「未見 r38 條目、可能沿用 JP6.2 URL」。實際讀 `config.py` 後**修正為更精確也更嚴重的結論**：不是「找不到→報錯」，而是 `>=` 級聯**沒有版本上限**，r38 會**靜默命中最頂端的 JP6.2 DS8.0.0 分支**，裝成「CUDA 12.6 的 DeepStream 落在 CUDA 13 環境」的假成功。決策從「擔心它抓不到」轉為「**擔心它抓到但抓錯，且看起來像對的**」——這反而強化了 `preflight.py`（用 `import pyds`/`deepstream-app --version` 驗真）的必要性。

（實機那端若再有轉折，請接續往下記。）

---

## 5. 怎麼跑（實機 Jetson 上）

```bash
# 0) 取得並安裝 jetson-containers（若還沒）
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh

# 1) 建置組合容器（DeepStream + PyTorch）
./build.sh                 # 內部：jetson-containers build --name=ds-torch-webcam deepstream pytorch torchvision

# 2) 進容器並跑健檢（= 依賴完整性驗收 AC-1）
./run.sh preflight         # 跑 app/preflight.py

# 3) 跑小應用（AC-2）
./run.sh app               # 跑 app/app.py，預設讀 /dev/video0，headless 也能跑

# 手動進容器 shell：
./run.sh shell
```

環境變數（可覆寫）：
- `VIDEO_DEVICE`（預設 `/dev/video0`）
- `CONTAINER_NAME`（預設 `ds-torch-webcam`）
- `INFER_EVERY`（每幾幀跑一次 PyTorch，預設 15）

---

## 6. 待辦 / 給下一棒的人

- [ ] 實機 `git pull` 最新 jetson-containers，確認 DEP-1（DS8.0 對 JP7.1 的 URL）。
- [ ] 跑 `./run.sh preflight`，把 §3 表格狀態更新為 🟢/🔴。
- [ ] 若 DEP-1 為 🔴：改用 NVIDIA NGC 的 DeepStream 8.0 JP7 容器當 base，PyTorch 用 pip/wheel 疊上去（備援方案，見 §7）。
- [ ] 確認 webcam 型號走 `v4l2src` 還是 CSI（`nvarguscamerasrc`），必要時調 `app/app.py` 的 `SOURCE_KIND`。

## 7. 備援方案（Plan B，若 jetson-containers 組合失敗）

若 `build.sh` 因 DEP-1 失敗，senior 的退路：
1. 以 NVIDIA 官方 `nvcr.io/nvidia/deepstream:8.0-*-triton-multiarch`（JP7/Thor 對應 tag）為 base。
2. 容器內 `pip install` 對應 CUDA 13 的 PyTorch wheel（NVIDIA Jetson PyPI index）。
3. 其餘應用碼（preflight.py / app.py）**不動**即可沿用 —— 這也是把應用邏輯與環境解耦的好處。
