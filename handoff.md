# Handoff — Jetson DeepStream + PyTorch + Webcam 小應用

> 目的：用 [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers) 在 **JetPack 6.2**（L4T r36.4.3 · CUDA 12.6 · Ubuntu 22.04 · Python 3.10）上，跑一個
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
- **平台決策（T8）**：目標已從 JP7.1 **改為 JP6.2**。JP6.2 = L4T r36.4.3，是 `deepstream/config.py` 明確對應 DS 8.0.0 的那一筆 → **原本的最大風險（DEP-1 樂觀誤配）直接解除**。詳見 §3、§4-T8。

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
| 6 | DeepStream 版本 | `deepstream-app --version` 可執行（JP6.2 → jetson-containers 選 DS 8.0.0 / pyds 1.2.2）|

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
| DEP-1 | **DeepStream 8.0 對目標平台的 jetson-containers 打包** | 🟢 **（改用 JP6.2 後解除）** | **原風險（JP7.1）**：選版是一串 `>=` 由高到低的 if-elif，最上面是 `L4T_VERSION >= 36.4.3 → DS 8.0.0 / pyds 1.2.2 / tarball 'deepstream_sdk_v8.0.0_jetson.tbz2'`。JP7.1 是 L4T **r38**，`38 >= 36.4.3` 成立 → 會**樂觀地**命中這筆「為 JP6.2 而寫」的建置，靜默誤配（見 §4-T6）。**現況（JP6.2）**：JP6.2 = L4T **r36.4.3**，正是這筆條目**原本就針對的版本** → 命中它是**精確且已驗證的相容組合**，不再是外推。風險消失。**唯一殘留提醒**：若日後又想上 JP7，本坑會重現，屆時走 §7 備援（NVIDIA NGC 官方 DS8.0 JP7 容器）。實機仍以 `preflight.py` 的 `import pyds` / `deepstream-app --version` 做最終確認。 |
| DEP-2 | **pyds（DeepStream Python bindings）** | 🟡 | JP7 對應 pyds 版本較新（DS8.0 → pyds ≥1.2.x）。jetson-containers 的 deepstream 包會在版本 ≥1.2.0 時**從源碼編譯** gst-python + bindings（需 meson/cmake/pybind11）。編譯鏈缺任一都會失敗。preflight 會直接 `import pyds` 抓這個。 |
| DEP-3 | **PyTorch on CUDA 13 / JP7** | 🟡 | jetson-containers 官方已宣稱支援 JetPack 7 (CUDA 13.x)。風險在 torchvision 版本需與 torch 對齊，否則 `import torchvision` 會因 C++ ABI 不合而炸。**應對**：app 對 torchvision 匯入做 try/except，失敗時降級為「不做分類、只驗證擷取路徑」。 |
| DEP-4 | **Python 版本落差** | 🟡→較低風險 | JP6.2 基底 Ubuntu 22.04 → **Python 3.10**（比 JP7.1 的 3.12 成熟、wheel 覆蓋更廣，這也是選 6.2 的附帶好處）。pyds/torch/torchvision 需與 3.10 對齊。preflight 印出 `sys.version` 方便比對。 |
| DEP-5 | **GStreamer nv 外掛註冊** | 🟡 | 容器內 `nvvideoconvert`/`nvstreammux`/`nvinfer` 需在 `GST_PLUGIN_PATH` 註冊。DeepStream install.sh 會處理，但若 base image 換過可能漏。preflight 用 `gst-inspect-1.0` 逐一檢查。 |
| DEP-6 | **webcam 裝置直通** | 🟡 | 容器要 `--device /dev/video0`。`jetson-containers run` 預設不一定帶。`run.sh` 已顯式加上 `--device` 與 X11 掛載。 |
| DEP-7 | **顯示 / X11** | 🟡 | 無螢幕或 headless 時 `nveglglessink`/imshow 會爆。app 預設 **headless 友善**：無 DISPLAY 就只印文字，不開視窗。 |
| DEP-8 | **Docker / NVIDIA container runtime 前置** | 🟢(設計已處理) | 正常 JetPack **已預裝** Docker + nvidia runtime；jetson-containers 的 `install.sh` **不裝 Docker**。`build.sh` 步驟 0 做 idempotent 檢查：缺 Docker 才裝、缺 nvidia runtime 才裝。**JP6.2 定案後**：nvidia runtime 從「只警告」改為 **apt 安裝 `nvidia-container-toolkit`**（L4T repo 已預配，套件明確，不再有裝到 x86 版的歧義；見 T9）。可用 `SKIP_DOCKER_SETUP=1` 跳過整段。 |
| DEP-10 | **`docker buildx` 建置失敗 exit 125 → 根因 = 用了 docker.io** | 🟢 **已確診並修正** | 實機 `./build.sh` 在**第一個 `build-essential` 階段**就 `docker buildx build ... exit 125`（base 還是乾淨的 `ubuntu:22.04`）。**實機確認**：`docker buildx version` → `unknown command`（buildx 根本沒裝）。**根因確定**：DEP-9 為避開 no-curl 而選 `apt install docker.io`，**Ubuntu 的 docker.io 只給 engine+CLI，不含 buildx/compose 的 cli-plugins**，而 jetson-containers 寫死 `docker buildx build` → 一呼叫就 125。這是「為省事繞過 curl」的錯誤取捨代價。**修法（build.sh 步驟 0 已改為正規流程，見 T10）**：先 `apt install curl` → 移除 docker.io → 從 **Docker 官方 apt 源裝 docker-ce 全套（含 buildx）** → `nvidia-container-toolkit` + `--set-as-default`。同時 `default-runtime` 已在實機手動設為 nvidia，步驟 0 也會確保。|
| DEP-11 | **r36.5 (JP6.2.1) 無預建 deepstream → 逼出源碼編譯；用 `--base` 降級融合解決** | 🟢 已解（實測） | autotag 明講 `Couldn't find a compatible container for deepstream`（L4T 36.5.0）。dustynv 的 deepstream 在 r36 線只到 **r36.2.0**（無 r36.4）。直接 `build deepstream pytorch` 會從 `ubuntu:22.04` 編整條 **19 層**（實測光 vulkan 一階 ~2hr）。**解法**：`--base=dustynv/deepstream:r36.2.0` 當地基，jetson-containers 只疊 pytorch。r36.2.0（DS 6.4.0 / cu122 / JP6.0 DP）在 r36.5 host 上**實測**：`pyds 1.1.10` ✅、`deepstream-app 6.4.0` ✅、5 個 gst nv 外掛全 ✅（靠 CUDA 驅動向前相容）；容器內讀到的 L4T 是 host 的 r36.5（掛載）。最新替代：NVIDIA 官方 DS7.1（cu126）當基底，需 `docker login nvcr.io`。**融合方式見 T12**：`jetson-containers build --base` 實測**不可行**（會重建 CUDA），改用薄 Dockerfile `FROM <deepstream> + pip install torch`。 |
| DEP-12 | **jetson wheel 索引域名搬家：`pypi.jetson-ai-lab.dev` 已死 → 改 `.io`** | 🟢 已修 | 實機 build 在 pip 步驟 `Name or service not known`。從開發機驗證：`.dev` 域名 **NXDOMAIN**（整個域名消失，非網路問題）；`pypi.jetson-ai-lab.io` 存活且 `jp6/cu126` 有 torch 2.9–2.11 的 cp310 aarch64 wheel（對齊 DS7.1 容器的 Python 3.10 / CUDA 12.6）。Dockerfile 預設已改 `.io`。**教訓**：社群 wheel 站會搬家，錯誤訊息是 DNS 類（NXDOMAIN）時先在另一台機器驗證域名，區分「站死了」vs「本機網路壞」。 |
| DEP-13 | **官方 DS7.1 容器缺 torch wheel 的系統庫（libopenblas）** | 🟢 已修（待實機過關） | torch 2.11.0 從 `.io` 索引安裝成功，但 `import torch` 炸 `libopenblas.so.0: cannot open shared object file`。jetson torch wheel 動態連結 OpenBLAS；dustynv 容器預裝了它，**NVIDIA 官方 DeepStream 容器沒有**（官方容器只保證 DeepStream 自身依賴）。修法：Dockerfile 補 `apt-get install libopenblas0`，並順手裝 numpy（app 需要）。**結構改進**：把「pip 裝 wheel（大下載）」與「補系統庫+驗證」拆成兩層——之後再缺其他 `.so` 只重跑小層，不重下載 torch。**教訓**：跨生態混搭（NVIDIA 官方 image + dustynv wheel）時，wheel 隱含的系統庫依賴不會自動跟過來，要逐一補。**續集**：openblas 補完後下一個缺 `libcudss.so.0`（cuDSS）。查 jetson-containers 的 `install_cudss.sh`/`install_cusparselt.sh` 得到正解：**加 NVIDIA CUDA 網路 apt 源（cuda-keyring ubuntu2204/arm64）→ `apt install cudss libcusparselt0`**，裝完即purge keyring（同 jetson-containers 做法）。已把 cuDSS + cuSPARSELt 一次補進層2（cuSPARSELt 是預判 torch 2.11 也會要）。層2迭代只花 ~50s，驗證了拆層設計。 |
| DEP-9 | **JP6.2 base 映像缺 curl（實機回報）** | 🟢(已修) | 使用者實機為 **JP6.2 base，無 docker 也無 curl**。原 `build.sh` 用 `curl get.docker.com` 裝 Docker → **在無 curl 的機器上直接失敗**。**修正**：步驟 0 全面改走 **apt**（base 一定有 apt），`docker.io` 裝 Docker、`nvidia-container-toolkit` 裝 runtime，**完全不依賴 curl**；另加 `ensure_base_tools` 把缺的 `curl`/`git` 補上。教訓：**不要假設 base 映像有 curl/wget，唯一能假設的是 apt。** |

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
- **T7 Docker 前置的取捨（新增 build.sh 步驟 0）**：被問「要不要自己裝 container」。查證 —— **JetPack 預裝 Docker + nvidia runtime，jetson-containers `install.sh` 不裝 Docker**，所以標準情況不用。但仍加了 idempotent 安裝：**缺 Docker 才裝**。關鍵決策是 **nvidia runtime「只警告不強裝」**——因為 Jetson 的 runtime 來源與 x86 `nvidia-container-toolkit` 不同，腳本亂裝會弄壞 JetPack，寧可停下來讓人確認。體現原則：**破壞性 / 系統級操作要保守，能 idempotent 就 idempotent，沒把握就不要自動化。**
- **T14 基底切到 NVIDIA 官方 DS 7.1（推翻 T13 的 r36.2.0 預設）**：使用者兩點質疑都成立——(1)「r36.2.0 不是 for 我的 36.5」：對，那是 JP6.0 DP 的產物，靠向前相容硬撐；(2)「看官方有沒有 release」：查證確認 **DeepStream 7.1 就是官方對 JP6.2 / L4T 36.4–36.5 的正式 release**，容器 `nvcr.io/nvidia/deepstream:7.1-triton-multiarch`（cu126，與 host 原生對齊）。另確認**沒有任何現成單一 image 同時含 DeepStream+PyTorch**（官方兩者分開出），所以「自己合一顆」不可避免——但維持零編譯：`FROM` 官方 DS7.1 + pip 裝 jetson-ai-lab **jp6/cu126 預建 torch wheel**（索引已設為 Dockerfile 預設）。同時把 host 設定拆成一次性 `setup-host.sh`，`build.sh` 只管做 image；**jetson-containers 工具本身已完全不需要安裝**（純 docker build/run）。代價：拉官方 image 需 `docker login nvcr.io`（NGC 免費帳號）。r36.2.0 降為免登入備援（`DEEPSTREAM_BASE=... TORCH_INDEX= ./build.sh`）。待實機驗證點：官方 DS7.1 容器內 **pyds 是否內建**（論壇稱含 pyds 1.2.0）——preflight 第一項就會揭曉，若缺再於 Dockerfile 補裝 pyds wheel。
- **T12 推翻 T11：`jetson-containers build --base` 不可行 → 改薄 Dockerfile + pip（真正定稿）**：實機跑 `--base` 融合,結果 **10 stages、在 `packages/cuda/cuda` 階段 `exit 100` 失敗、耗 54 分**。學到 `--base` 的真實語意:它只把 deepstream 的子樹省掉(19→10),但 **pytorch 仍會把「自己的」依賴(含 CUDA)在基底上從源碼重建**,即使基底已有 CUDA——jetson-containers **不會 introspect 基底內容**。所以「用 jetson-containers 融合」在這裡達不到「快速」。**最終定稿**:薄 Dockerfile `FROM <現成 deepstream> ` + `pip install torch torchvision`(預建 wheel),`build.sh` 改用 `docker build`。這正是使用者兩輪前質疑為「土」、但**實測證明才是對的**做法——現成 deepstream 已含 CUDA,pip 只補 torch,零重建。關鍵優點:若 pip 找不到 wheel,**秒級失敗**,不再賭 54 分。教訓:**別再假設工具的 flag 語意,實測為準;我這次也確實連錯兩版(T11 的 --base 是誤判)。**
- **T13 使用者拍板：CUDA 12.2 / r36.2.0**：確認要求「CUDA 也要用 jetson-containers 提供的」——澄清後確認**現行做法已滿足**（基底 `dustynv/deepstream:r36.2.0` 的 CUDA 就是 jetson-containers 打包的，pip torch wheel 也對著它編）。使用者在「12.2 全 jetson-containers 快速可用」vs「12.6 最新但要數小時源碼 build」間選 **12.2**。定案不變，程式碼不動。（cu126 需求若日後出現，走源碼 build，先查磁碟空間——`exit 100` 疑為空間不足。）
- **T11 定稿：`--base` 融合取代源碼編譯（build/run 定版）**：實測揭露兩件事 —— (1) r36.5 沒有預建 deepstream，原 `build deepstream pytorch` 會從 `ubuntu:22.04` 把 19 層全編、數小時；(2) 但把 `dustynv/deepstream:r36.2.0` 降級到 r36.5 host 上跑，pyds/gstreamer 全過。決策定稿：**`build.sh` 改用 `jetson-containers build --base=<現成 deepstream> pytorch torchvision`**——deepstream 用現成、只疊 pytorch wheel，幾分鐘。**`run.sh` 改為明確 `docker run`**，棄 `autotag`/`jetson-containers run`（避開 sudo 汙染造成的權限雷 + 不必要的重建）。基底預設 r36.2.0（免 NGC 登入、已實測），最新選項留 NVIDIA DS7.1（`DEEPSTREAM_BASE=nvcr.io/...` + `CUDA_VERSION=12.6`）。**兩個教訓**：(a) jetson-containers 的融合本身沒問題，慢是因為「沒預建就從零編」——餵它現成 `--base` 才是正確快速用法；(b) **jetson-containers / docker 一律不要用 sudo 跑**（會把 repo 弄成 root 擁有，後續非 sudo 執行全卡 EPERM），正解是 `newgrp docker` / 重新登入讓 docker 群組生效。
- **T10 修正 Docker 安裝：回到正規官方流程（推翻 T9 的 docker.io 取捨）**：T9 為了「base 沒 curl」改用 `apt install docker.io`——**這是把簡單問題複雜化的錯誤決策**：沒 curl 就 `apt install curl` 就好，卻繞去 docker.io，反而引入 buildx 缺失（DEP-10 的 exit 125）。使用者一針見血指出後修正。決策改為：**步驟 0 一律走 Docker 官方流程**（補 curl → 移除 docker.io → 官方源裝 docker-ce 全套含 buildx → nvidia toolkit + default-runtime）。且依使用者要求「完整重裝、不要 fallback」，**移除所有「偵測到就跳過/降級」的分支**，改為線性流程 + `set -e`：任一步真失敗就停、不繞路。教訓：**（1）不要為了規避一個小缺件（curl）而換掉整個正確方案（官方 Docker）；（2）jetson-containers 需要 buildx，Docker 必須從官方源裝，docker.io 不合格。**
- **T9 實機是 JP6.2 base，無 docker 無 curl（修 build.sh）**：實機回報後發現 base 映像**連 curl 都沒有**，原本用 `curl get.docker.com` 的裝法在這台直接失敗。決策：**步驟 0 全面改 apt**（`docker.io` + `nvidia-container-toolkit`），零 curl 依賴，並補 `ensure_base_tools` 補齊 curl/git。同時因為平台已鎖 JP6.2 + 是 JetPack 裝置（L4T apt 已預配），nvidia runtime 從 T7 的「只警告不裝」**升級為「apt 主動安裝」**——先前的保守是因為平台未定、怕裝錯來源；平台一確定，歧義消失，就該把它自動化。體現原則：**保守程度應隨「不確定性」動態調整，不是一味保守。** 教訓也記進 DEP-9：**base 映像唯一能假設的是 apt。**
- **T8 平台改用 JP6.2（重大決策，解除 DEP-1）**：使用者決定安裝 **JetPack 6.2 base**，目標平台從 JP7.1 下修。這不是退步而是**風險工程**：JP6.2 = L4T r36.4.3，正好是 `config.py` 明確支援 DS 8.0.0 的那一筆 → T6 發現的「樂觀誤配」風險**直接消失**，DEP-1 由 🔴 轉 🟢。附帶好處：Ubuntu 22.04 / Python 3.10 生態比 24.04 / 3.12 成熟，wheel 覆蓋更廣（DEP-4 降風險）。代價：不是最新的 CUDA 13 / Thor 特性——但本任務只為驗依賴，**穩定 > 最新** 是對的取捨。程式碼（preflight/app/run.sh）**完全不用改**：版本無關 + `autotag` 自動挑 image，這正是當初把應用邏輯與平台解耦的回報。只更新了 README/build.sh 的平台標示與本決策紀錄。
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
