# 本機會議錄音轉錄工具

所有錄音與逐字稿均於本機處理。程式不會主動將音訊上傳到任何第三方服務。
使用 faster-whisper / CTranslate2，無需 API Key、不使用 OpenAI API 或付費語音服務；只產生逐字稿，不自動生成摘要。

## 桌面介面：拖入音檔即可轉錄

已完成安裝後，雙擊 **`run-ui.bat`** 開啟桌面視窗。舊版使用者先重新執行一次 `setup.ps1`，uv 會自動補齊介面與說話者模型執行套件。

1. 把音檔拖到視窗，或按「選擇檔案」。可以一次加入多個錄音。
2. 選擇辨識模型；預設 `medium`，一般 CPU 若希望較快可選 `small`。
3. 需要標示不同人時，勾選「區分不同人說話」。人數可選「自動」，或指定 1–20 人。
4. 按「開始轉錄」。介面顯示目前階段、百分比、已處理音訊長度、耗時與**本階段預估剩餘時間**。
5. 文字會陸續出現在預覽區；完成後可複製全文或開啟輸出資料夾。勾選說話者區分時，分析完成後預覽及輸出會更新為 `【說話者 1】…`、`【說話者 2】…`。

按「停止並保留進度」後，程式會在目前辨識步驟返回後停止；下次按開始可續跑。模型載入或音訊解碼期間的停止可能需要等待，介面仍可操作。處理中關閉視窗會詢問是否停止，並等待背景工作結束。

預估時間根據目前階段的實際處理速度計算，會扣除續跑時已有的進度。首次模型下載、載入及剛開始辨識時顯示「估算中」，不提供假進度；開啟說話者區分時會明確分成「1/2 辨識語音」及「2/2 分析不同說話者」。預估值不包含尚未開始的階段或後續檔案，靜音比例與裝置速度改變時也會變動。

也可以從 PowerShell 啟動：

```powershell
uv run --no-sync python gui.py
# 或直接帶入檔案，打開介面後再按開始：
.\.venv\Scripts\python.exe gui.py "C:\Recordings\會議.m4a"
```

`run.bat` 保留原本的命令列批次模式；`run-ui.bat` 為新桌面介面。

## 本機說話者區分

- 使用公開的中文 3D-Speaker ERes2Net 聲紋特徵模型與分群，第一次需要下載約 40 MB 模型，存於 `models/speakers`。不需要帳號、Token 或 API Key；快取後可以離線執行。模型來源與用法見 [sherpa-onnx 官方範例](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py)。
- 先在音訊中偵測語音，取最多 3 秒的重疊視窗分析聲音特徵，再在整份錄音中分群。說話者分析使用較短的靜音門檻及前後留白，不跨已偵測的靜音合併視窗；不影響語音辨識的 VAD 設定。長句末尾維持完整視窗，減少短尾段的聲紋波動。
- 自動人數優先由至少 1.5 秒的片段建立群組，再以較嚴格的距離門檻合併接近的群組中心，降低同一人因語氣改變被拆開的機會。較短片段比對既有群組，證據不足時標為「未能區分」，不另算新人；若整份錄音只有短句，則仍以短句估計並提示可靠性較低。只說短句的新說話者可能無法辨認，聲音相近的人仍可能被合併；已知人數時可指定，停用此自動合併與短句篩選。
- 同一位說話者再次發言時共用編號。編號依首次出現順序排列，不會推測真實姓名或身分。Whisper 的逐字時間戳以重疊語音視窗加權投票對應說話者，分數接近或沒有覆蓋時標為「未能區分」。五種輸出均保留說話者標記；JSON 另有 `speaker`、`speaker_alignment`、原始文字及時間戳。
- 分析每次只解碼最多 5 分鐘音訊，保存小型聲音特徵及分析 checkpoint；最多取 2,000 個代表視窗做分群，避免長錄音的距離矩陣無限增長。特徵與結果都留在本機 `output`。
- 說話者分析使用獨立 CPU 程序，並明確載入專案內的 ONNX Runtime，避免 Windows 系統舊 DLL 干擾主轉錄程序。分析失敗時，已完成的純逐字稿與轉錄 checkpoint 仍會保留。
- 切換是否區分說話者或變更人數，會重用已完成的語音辨識及適用的分析快取。早期版本的 checkpoint 沒有逐字時間戳時，只能按舊片段標示；勾選「重新轉錄」可取得逐字對齊，舊結果會先備份。
- 說話者演算法更新後，再次勾選「區分不同人說話」並按開始，即會重用逐字稿、更新說話者分析；不需勾選「重新轉錄」。舊版聲紋快取會保留，但不混用於新版分析。
- 自動估計人數、短句、相似聲音、噪音和同時說話仍可能分錯；已知人數時可直接指定。此功能不會把重疊的多人聲音分離成獨立音軌，也不會補回語音辨識漏掉的句子。

命令列亦支援：

```powershell
uv run --no-sync python transcribe.py "會議.m4a" --diarize --num-speakers 2
```

設定檔預設值為 `diarize: false`、`num_speakers: 0`、`speaker_threshold: 0.45`。自動人數時，較小的 threshold 傾向分成較多人；只有確定需要調整時才修改。桌面選項只作用於本次執行，不會自動改寫 YAML。

## 第一次安裝（Windows 10 / 11）

1. 將此專案資料夾放在自己的電腦，例如 `C:\Tools\meeting-transcriber`。路徑及錄音名稱可以含中文、空白。
2. 在專案資料夾空白處按右鍵，開啟終端機或 PowerShell，貼上這一個指令（不需預先安裝 Python）：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

若從 Git 取得此專案，先執行 `git clone <實際儲存庫網址>`，再 `cd <下載的專案資料夾>`；本專案未預設任何遠端儲存庫網址。

安裝統一由 **uv** 管理，會自動完成：

1. 找到已安裝的 uv；若沒有，從 [uv 官方安裝來源](https://docs.astral.sh/uv/getting-started/installation/) 安裝至專案 `.tools/uv`。
2. 準備 Python 3.12 至 `.python`，不影響系統 Python。
3. 依 `uv.lock` 建立或同步 `.venv`，包含主程式及 pytest。
4. 檢查 FFmpeg、建立資料夾，執行離線 self-test。

整個流程不需要系統管理員權限，不修改系統 PATH，也不執行 `ensurepip` 或 `pip install`。中斷後重跑同一個指令即可。已有套件會按鎖定版本重用；套件版本以 `pyproject.toml` / `uv.lock` 為準，`requirements.txt` 只供相容用途。

FFmpeg 由 `imageio-ffmpeg` 的 Windows 套件隨附，通常不用額外安裝。首次準備 uv、Python、套件與模型需要網路；**模型下載完成後可以完全離線轉錄**。不同大小的模型需要各自下載。

若電腦政策禁止執行 PowerShell，請遵循公司的軟體安裝程序。`-ExecutionPolicy Bypass` 僅用於此次程序，不修改系統政策。

## 日常使用

1. 把 `.m4a` 等錄音放入 `input` 資料夾。
2. 雙擊 **run.bat**。
3. 等待畫面顯示 `Transcription completed`。
4. 開啟 `output\錄音名稱\錄音名稱.md`，可交給 AI 整理會議記錄。

預設依序處理 input 第一層所有錄音。已完成且來源及辨識設定未變更的檔案會顯示 `[SKIP]`。多檔批次中某檔失敗會保留結果並繼續其他檔案；只要有失敗，結束碼就是 1。Ctrl+C 中斷的結束碼為 130。

支援 `.m4a .mp3 .wav .flac .ogg .mp4 .mov .webm .aac .wma .mkv .opus .aiff .aif .mpeg .mpg`。明確指定單一檔案時也會嘗試解析其他 FFmpeg 支援的格式。影片只處理第一條音軌。

## 指定錄音與命令列選項

使用專案自己的 Python，不需要啟動虛擬環境：

```powershell
.\.venv\Scripts\python.exe transcribe.py "C:\Recordings\新錄音.m4a"
.\.venv\Scripts\python.exe transcribe.py "C:\Recordings\新錄音.m4a" --model medium
.\.venv\Scripts\python.exe transcribe.py input --recursive
.\.venv\Scripts\python.exe transcribe.py input --offline
.\.venv\Scripts\python.exe transcribe.py input --force
.\run.ps1 "C:\Recordings\新錄音.m4a"
```

若已執行 `.\.venv\Scripts\Activate.ps1`，也可直接用 `python transcribe.py "錄音.m4a"`。雙擊 `run.bat` 和不指定輸入參數時，皆遵循 `config.yaml` 的 `input_dir`。

已完成 setup 且 uv 在 PATH 時，也可以使用 `uv run --no-sync python transcribe.py input`；`--no-sync` 使用已準備好的環境，不在轉錄時同步套件。

```text
--model small|medium|large-v3     模型名稱，也可指定本機 CTranslate2 模型目錄
--language zh                   語言代碼；預設中文
--device auto|cpu|cuda           auto 偵測 GPU；CUDA 失敗會回退 CPU
--compute-type int8             覆寫運算精度；auto 為 GPU float16 / CPU int8
--config C:\path\config.yaml    使用另一份設定
--output-dir C:\Transcripts     覆寫輸出目錄
--recursive                     包含子目錄
--offline                       模型缺少時報錯，完全禁止下載
--force                         備份舊結果後重新辨識
--download-model                只準備模型，不處理錄音
--self-test                     離線套件、FFmpeg、中文路徑與轉換檢查
--help                          顯示完整選項
```

YAML 內的相對路徑以 **config.yaml 所在目錄**為基準；CLI 輸入檔案與 `--output-dir` 以目前工作目錄為基準。

## 模型、CPU 與 GPU

修改 `config.yaml` 即可：

```yaml
model: medium
language: zh
device: auto
compute_type: auto
```

- 一般 CPU：`small` 較快；`medium` 為預設，較有利於混合中文與英文術語，但長錄音可能比音訊時長更慢。
- NVIDIA GPU：可使用 `medium` 或 `large-v3`；實際速度與可用記憶體依硬體和錄音而異。
- CTranslate2 偵測 CUDA、檢查精度支援並嘗試載入。載入或 Python 可捕捉的推論錯誤會自動回退 CPU int8，保留已寫入片段。
- 現行 faster-whisper 的 GPU 環境需要相容的 NVIDIA 驅動、CUDA 12 cuBLAS 與 cuDNN 9；依 [faster-whisper 官方安裝說明](https://github.com/SYSTRAN/faster-whisper#gpu) 準備。沒有這些套件仍可用 CPU。程式不安裝顯示卡驅動。

先下載模型，再在無網路環境使用：

```powershell
.\.venv\Scripts\python.exe transcribe.py --download-model --model medium
.\.venv\Scripts\python.exe transcribe.py input --offline
```

模型放在 `models`，採 Hugging Face 預設快取結構。每次先確認完整本機快取，有快取就不查詢網路。自訂模型目錄必須至少包含 `model.bin`、`config.json`、`tokenizer.json`；建議攜帶完整下載目錄。未包含 tokenizer 的模型會直接報錯，不會暗中從網路補抓。

## 長錄音、保存與續跑

faster-whisper 的 segment iterator 雖逐段回傳文字，直接傳入完整檔案仍會解碼整段音訊。因此本工具先以 FFmpeg 擷取 **300 秒**的 16 kHz 單聲道音訊，再使用 iterator 辨識；每次最多保留一個音訊區段，沒有只處理前 60 秒的限制。參考 [faster-whisper 原始碼](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)。

每一段文字立即追加至 `transcript.partial.jsonl`，flush + fsync 後才更新記憶體狀態。已辨識的文字只佔很少記憶體；音訊不會整段載入 RAM。每個區段另原子更新 `checkpoint.json`，因此長時間靜音也能保留處理進度。

中斷後重新執行相同命令，會從最後已完成時間前 **15 秒**接續。重疊部分依絕對時間及 word timestamps 去重，不對語句做模糊比對。若最後一次 JSONL 寫入遭截斷，會保留 `.damaged-*` 備份並使用之前完整的片段；中間行損毀則停止，不默默刪除資料。

輸出原子替換，全部格式成功寫完才標記 completed。`--force`、來源 size / mtime 變動或辨識設定變更時，舊結果會移到該輸出目錄的 `history`；不會直接刪除。輸出格式開關變更不必重辨識，缺少的格式可由 partial 再輸出。同名但來源路徑不同的錄音使用 `名稱--路徑雜湊` 避免相互覆蓋。同一 output 目錄同時僅允許一個轉錄程序。

可調整：

```yaml
chunk_seconds: 300       # 30–900 秒，減小可降低音訊暫存記憶體
overlap_seconds: 15      # 0–30 秒，且不得大於 chunk_seconds 的一半
vad_filter: true
vad_parameters:
  min_silence_duration_ms: 500
  speech_pad_ms: 400
  threshold: 0.5
```

## 輸出格式與原始文字

```text
output/新錄音/
  新錄音.txt                 純文字，適合複製貼上
  新錄音.timestamps.txt      每段附 HH:MM:SS
  新錄音.srt                 標準毫秒時間字幕
  新錄音.json                metadata、segments、原始文字及可疑標記
  新錄音.md                  按 10 分鐘分組，附 AI 整理提示
  metadata.json              來源識別、模型、執行裝置、完成狀態
  transcript.partial.jsonl   每段即時保存，完成後也保留
  checkpoint.json            含靜音時間的已處理進度
```

`text` 為可讀取文字；`raw_text` 保留未校正的已採用文字；`original_segment_text` 保留模型原始 segment 文字（重疊裁切前）。繁體轉換採 OpenCC `s2twp`，可透過 `traditional_chinese: false` 關閉。

若錄音名稱恰為 `metadata` 或 `checkpoint`，五種輸出會加上 `.transcript`（例如 `metadata.transcript.json`），避免覆蓋同目錄的控制檔案。

`terminology` 只依設定做明確、大小寫不敏感的替換；英文詞檢查 ASCII 詞界，不把 `wafer123`、`erp_id` 或 `superpcb` 中的片段擅自換掉。沒有 fuzzy matching。`initial_prompt` 完全來自 YAML，可依會議主題修改。

重複語句、低信心與可能的字幕宣傳幻覺會在 JSON 中標記 `suspicious` 與原因，**仍保留全文**，不自動刪除或改寫。VAD 能減少靜音誤辨，但不是內容正確性的保證。

## 測試與疑難排解

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe transcribe.py --self-test
```

使用 uv 的等效測試指令為 `uv run --no-sync pytest`。如需更新套件，先修改 `pyproject.toml`，執行 `uv lock` 更新鎖定檔，再重跑 setup；一般使用者不需要更新鎖定檔。

測試涵蓋格式、繁體與術語、設定、來源去重、原子保存、中文路徑、實際 FFmpeg m4a 解碼、模擬 120 分鐘區段流程、中斷續跑、靜音恢復及 GPU 失敗回退。單元測試不下載 Whisper 模型。

- `找不到輸入檔案`：確認路徑，用雙引號包住含空白的路徑。
- `FFmpeg 尚未安裝`：重新執行 setup，或在 `ffmpeg_path` 指定 `C:/ffmpeg/bin/ffmpeg.exe`。也可自行安裝 FFmpeg 並加入 PATH。
- `無法解析音訊格式`：確認檔案有音軌且可播放，檢查是否損毀或音軌提早結束。
- `Whisper 模型下載失敗`：確認網路、代理與磁碟空間，或使用已備妥的本機模型目錄。
- `CUDA 不可用`：警告而非失敗；會繼續使用 CPU。
- 詳細錯誤堆疊只寫入 `logs/transcribe-YYYYMMDD.log`。log 含本機路徑等資訊，分享前請自行檢查。
- 另一程序正在使用 output：等待該程序結束；鎖會由作業系統自動釋放，不需刪除鎖檔。

## 已知限制

- 語音辨識不是人工校對；噪音、多人同時說話、術語和區段邊界可能仍有漏字或重複。重疊去重依模型時間戳，並非逐字完美對齊。可選的說話者區分為近似標記，仍需人工確認。
- 無法從 CTranslate2 原生程序直接崩潰中即時切換 CPU；已 fsync 的 partial 仍保留，可下次加 `--device cpu` 接續。
- 記憶體除區段音訊外仍需容納模型；CPU medium 需要足夠 RAM，不足時請改 small。暫存解碼 WAV 存在輸出目錄並於正常離開時清除，硬性終止可能留下 `audio-*` 暫存目錄。
- 來源辨識用絕對路徑、大小和 nanosecond mtime，沒有逐 byte 雜湊；若自行修改檔案並刻意維持這些屬性，請用 `--force`。
- 系統當機的持久化程度也受磁碟與作業系統影響；請另行備份原始錄音。所有 partial/history 都保留，會佔用額外磁碟。
- 本專案不附使用者的 56 分鐘中文會議錄音；辨識品質與實際硬體速度須以自己的錄音驗收。測試結果及執行環境記錄見 `VALIDATION.md`。


## 本地語音辨識品質測試（免費、離線）

仍使用 Faster-Whisper，不需要任何雲端 API、Token 或訂閱費。低階 CPU 可維持 `small` 或 `medium` + CPU int8；GPU 不作為必要條件。首次下載開源模型需要網路，快取後可用 `--offline`。

- **自動辨識詞庫**：啟動時將 `config.yaml` 的 `vocabulary`（通用術語）、`terminology` 的正確寫法，以及本機的 `vocabulary.local.txt` 合併去重，自動送給 Faster-Whisper；每次打開 GUI 不必重填。
- **私人術語**：專案根目錄建立 `vocabulary.local.txt`，每行一個詞，可用 `#` 開頭寫註解。這個檔案已被 `.gitignore` 排除，不要使用 `git add -f` 將內部資料上傳到公開儲存庫。
- GUI「辨識關鍵詞」僅用於**本次額外補充**，例如 `iCube, Reel ID`；多字英文詞請以逗號分開，避免 `Reel ID` 被拆成兩個詞。
- 詞彙為模型的辨識提示，不是保證；詞表過長或包含不相關的詞也可能導致誤辨。原始 `raw_text` 仍保留。
- GUI「連貫上下文（實驗性）」預設關閉；僅在有人工校對樣本後嘗試啟用。此參數只影響每次呼叫模型的音訊區段內上下文，不會跨越 300 秒獨立視窗。
- CLI 亦可輸入 `--hotwords "iCube SMT 料站表"`、`--context`。
- 更動辨識參數會改變 checkpoint fingerprint。重新轉錄時先封存舊資料，不會靜默覆蓋。
- 預設 `initial_prompt` 已簡化，避免同時提示容易混淆且不一定出現的詞。

### 比較字元錯誤率 CER

自行聽寫一段 1–5 分鐘真實錄音，儲存為 `reference.txt`。用本程式輸出同段錄音的 JSON 後執行：

```powershell
python scripts/measure_cer.py reference.txt output/錄音檔名/錄音檔名.json
python scripts/measure_cer.py reference.txt output/錄音檔名/錄音檔名.json --raw
```

工具完全離線、只用標準函式庫。預設 NFKC 正規化、忽略空白及標點，並以編輯距離除以參考字數計算 CER。若語句涉及否定詞、數量、決議或爭議，仍須依原始錄音及時間戳人工核對。
