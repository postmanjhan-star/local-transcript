# 實作與驗證記錄

最近更新：2026-09-18。於本機 Windows / PowerShell 執行。

## UI 後續修正與桌面執行驗證

- 修正停止後移除檔案時，尚未完成的預覽文字沒有跟隨清單位置更新的問題；避免顯示其他錄音的內容。
- 修正輸出檔案無法讀取時仍保留上一份預覽的問題；現在會清空預覽、停用複製並顯示處理訊息。
- 最新完整測試：`uv run --no-sync pytest -q`，**76 passed in 16.46s**，含上述兩項回歸測試。
- 使用 `run-ui.bat` 所用的 `.venv\Scripts\pythonw.exe` 執行 `scripts/gui_smoke_test.py --model small`，退出代碼 0。31.209 秒雙人中文錄音在 CPU 完成耗時 28.42 秒；兩個說話者、交替四輪、五種輸出、Qt 完成狀態及各階段進度回報均通過。此驗證仍使用 Qt offscreen，並非實體滑鼠操作。

## 桌面 UI 與說話者區分（2026-09-18）

雙擊 `run-ui.bat` 開啟 PySide6 桌面介面，可拖入多個本機錄音、選擇模型、切換說話者區分、自動判斷或指定人數，並查看分階段進度、已耗時間、剩餘時間估算及逐字稿。取消後保留可續跑結果。

本次驗證結果：

- `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1`：uv 鎖定環境同步與 self-test 通過，包含 Qt 6.11.2；鎖定檔解析 46 個套件，本機安裝 44 個。
- `uv pip check --python .venv\Scripts\python.exe`：All installed packages are compatible。`uv sync --locked --check --offline`：Would make no changes。已明確納入 Windows wheel 要求的 `sherpa-onnx-core==1.13.8`。
- `uv run --no-sync pytest -q`：**74 passed in 17.05s**。新增 Qt 拖放事件、批次背景執行、切換檔案預覽、階段 ETA、說話者聚類與逐字對齊、ASR 快取重用、失敗保留文字稿、獨立子程序及取消續跑測試。
- `scripts/gui_smoke_test.py --model small`：使用實際 Qt 視窗及背景工作執行緒，以 CPU int8 處理 31.209 秒雙人中文 m4a；最終環境重跑完成時間 29.34 秒，五種輸出均存在，進度完成 100%。
- 測試錄音由 Windows Hanhan / Zhiwei 兩個離線合成聲音交替四輪。指定兩人時，各輪主要聲音標籤為 `[0, 1, 0, 1]`；再以自動人數執行，亦得到兩個說話者。修改人數重用既有 ASR 與聲紋 checkpoint。
- 說話者模型為本機 3D-Speaker ERes2Net Chinese；sherpa-onnx 1.13.8、onnxruntime 1.30.0、scikit-learn 1.9.1。Windows 子程序明確載入虛擬環境的 ONNX Runtime，避免系統舊 DLL 衝突。

GUI 測試使用 Qt offscreen 平台；已檢視實際視窗截圖及繁體中文字型，未聲稱測試了實體滑鼠拖曳。驗證產物位於 `.smoke/speakers/`，包含 `ui-ready.png`、`ui-completed.png`、`gui-validation.json`、測試音檔及逐字稿。

說話者區分是近似分析；本次實測仍有換人邊界的一個詞被分到相鄰說話者。尚未以真實多人會議或重疊對話驗收。ETA 依當次階段的實際速度估算；載入模型或尚無速度樣本時顯示估算中，不能當作精確完成時間。預設 medium 未改動；本機目前另有 tiny、small 與聲紋模型快取。

以下保留 2026-09-17 的 CLI / uv 基線驗證，套件數與 59 項測試數為當時記錄。

## uv 環境更新

安裝流程已改用 uv 0.12.5，新增 `.python-version`（3.12）與 `uv.lock`，測試套件納入 `dev` dependency group。`setup.ps1` 使用 `uv python install --no-bin 3.12` 與 `uv sync --locked --managed-python --python 3.12 --group dev`，不再使用 ensurepip / pip。

本次實際驗證：

- `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1`：成功同步目前環境，self-test 通過；修正後再次執行成功。
- `uv run --no-sync pytest -q`：**59 passed**。
- `uv run --no-sync python --version`：Python 3.12.14。
- `uv sync --locked --check --offline`：Would make no changes。
- `uv pip check --python .venv\Scripts\python.exe`：All installed packages are compatible。

目前安裝 34 個套件；鎖定檔共解析 35 個套件，包含不同 Python 版本的條件依賴。此機已有 uv，因此自動下載 uv 的分支未於本次觸發。下方 `python -m pip check` 記錄為遷移前驗證；遷移後請使用上列 `uv pip check`。

## 已完成

- 完整 Python 專案、虛擬環境、套件設定、YAML、Windows 安裝與啟動腳本。
- faster-whisper 本機辨識；CTranslate2 裝置偵測與 CPU int8 fallback。
- FFmpeg 有界音訊分段；逐 segment fsync 保存；重疊續跑與時間戳去重。
- TXT、時間戳 TXT、SRT、JSON、Markdown；保留原始文字與可疑片段標記。
- OpenCC 繁體轉換、安全術語替換、可配置 prompt / VAD。
- 批次處理、已完成跳過、變更偵測、force 備份、同名來源隔離、輸出程序鎖。
- Rich 進度、UTF-8 中文路徑與檔名、完整例外 log。

## 執行環境

| 項目 | 實測值 |
| --- | --- |
| Python | CPython 3.12.14，64-bit |
| faster-whisper | 1.2.1 |
| CTranslate2 | 4.8.2 |
| PyAV | 16.1.0 |
| FFmpeg | imageio-ffmpeg 0.6.0 隨附 Windows FFmpeg 7.1 |
| pytest | 9.1.1 |
| 實際辨識裝置 | CPU int8；CTranslate2 偵測到 0 個 CUDA 裝置 |
| smoke test 模型 | tiny，多語言；下載後所有辨識以 offline 執行 |
| 專案預設模型 | medium，未因測試而更動 |

## 自動測試

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

結果：**59 passed**。

涵蓋時間戳及毫秒進位、SRT、中文轉換、術語詞界、非串連替換、設定驗證、路徑、來源變更、同名衝突、保留檔名、截斷 checkpoint 修復、損毀資料保留、原子寫入失敗保留舊檔、程序鎖、120 分鐘完整區段流程、崩潰後續跑、靜音進度保存、CUDA 初始化與推論 fallback、輸出失敗後恢復、force 備份、轉錄中來源變動，以及 CLI 錯誤處理。

其中 FFmpeg 測試實際產生 **65 秒 m4a**，以中文含空白檔名讀取並擷取第 61 秒以後的音訊；其餘長時間／錯誤注入情境使用替身模型，以便離線快速重現。

其他檢查：

- `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1`：成功安裝套件及 editable project，self-test 通過。
- `python transcribe.py --self-test`：套件、FFmpeg、中文檔名、WAV 分段與繁體轉換通過，不下載模型。
- `run.ps1 --help`、`run.bat --help`：命令列參數成功傳入。
- `python -m compileall -q src transcribe.py scripts`：通過。
- `python -m pip check`：No broken requirements found。

## 真實本機語音辨識 smoke test

以 Windows 內建 **Microsoft Hanhan Desktop** 離線合成中文測試句，沒有使用網路語音合成。FFmpeg 將同一段語音放在音檔開頭及結尾，中間補靜音；輸出 AAC / m4a。

```powershell
.\.venv\Scripts\python.exe transcribe.py --download-model --model tiny
.\.venv\Scripts\python.exe scripts\smoke_test.py --source ".smoke\中文語音.wav" --duration 80
.\.venv\Scripts\python.exe scripts\smoke_test.py --source ".smoke\中文語音.wav" --duration 3360
```

| 驗證 | 80 秒音檔 | 56 分鐘音檔 |
| --- | --- | --- |
| 實際探測長度 | 80.000 秒 | 3360.000 秒 |
| 最後辨識時間戳 | 79.36 秒 | 3359.33 秒（55:59.33） |
| 首尾語音皆有輸出 | 通過 | 通過 |
| 時間戳單調且未超出音檔 | 通過 | 通過 |
| 五種 UTF-8 輸出 | 通過 | 通過 |
| metadata.completed | true | true |
| 再次執行跳過 | 通過 | 通過 |
| 辨識裝置 | CPU int8 | CPU int8 |

原始驗證產物保留於 `.smoke/real-80` 與 `.smoke/real-3360`，各含 `validation.json`、音檔、設定、log 和逐字稿。`.smoke` 不納入版本控制。可用自己的短語音 WAV 重跑 `scripts/smoke_test.py`；此測試需要已快取的指定模型。

## 驗證範圍與限制

- **56 分鐘測試的大部分內容是靜音**，用來檢查完整時間軸、分段和末段辨識；處理耗時不能代表 56 分鐘連續會議的速度。
- 沒有提供規格中的真實「新錄音.m4a」，因此沒有聲稱已驗收真實會議的術語正確率、多人對話品質或連續語音的 120 分鐘辨識效能。
- CPU 已實測；CUDA 初始化與推論錯誤切換以單元測試驗證，本機無 NVIDIA GPU，尚未進行 GPU 實機辨識。
- tiny 僅用於 smoke test，測試中仍有術語誤辨；正式預設保持 medium。首次使用 medium 仍會下載該模型，之後使用本機快取。
- 中斷恢復與磁碟失敗透過錯誤注入測試，未模擬實體斷電。原生函式庫直接終止程序時，下次以 `--device cpu` 可接續已保存結果。

## 現在使用

本工作目錄的 `.venv` 與套件已準備好。

1. 將錄音放入 `input`。
2. 雙擊 `run.bat`。
3. 首次等待 medium 模型下載及本機轉錄。
4. 開啟 `output/錄音名稱/錄音名稱.md`。

完整操作及調整方法見 [README.md](README.md)。
