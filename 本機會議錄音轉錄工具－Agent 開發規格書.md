# 本機會議錄音轉錄工具規格書

## 1. 專案目標

請建立一套可在 Windows 本機執行的「會議錄音自動轉錄工具」。

主要用途：

```text
.m4a / .mp3 / .wav / .mp4
        ↓
本機 Whisper 語音辨識
        ↓
逐字稿 TXT
字幕 SRT
結構化 JSON
        ↓
交給 ChatGPT / Claude / Codex
        ↓
整理成會議紀錄、需求書、規格書
```

核心要求：

- 不使用 AccurateScribe
- 不使用 OpenAI API
- 不使用任何付費雲端 Speech-to-Text API
- 不需要 API Key
- 音訊不可上傳雲端
- 所有語音辨識均於本機執行
- 優先支援繁體中文會議
- 能處理 30～120 分鐘長錄音
- 可直接處理 `.m4a`
- 產生完整逐字稿，而不是只有前 1 分鐘
- 即使長錄音，也要可靠完成
- 要有進度顯示
- 執行失敗不得導致已完成的結果消失

---

# 2. 使用環境

主要環境：

```text
OS: Windows 10 / Windows 11
Shell: PowerShell
Python: 3.11 或 3.12
```

可能執行於：

1. 公司一般辦公電腦
   - 可能只有 CPU
   - 可能沒有 NVIDIA GPU
   - 權限有限
   - 不希望需要系統管理員權限

2. 個人電腦
   - NVIDIA GPU
   - VRAM 16 GB
   - 可使用 CUDA

因此程式必須：

```text
GPU 可用 → 自動使用 GPU
GPU 不可用 → 自動 fallback CPU
```

不要讓使用者手動修改程式碼切換。

---

# 3. 技術選型

語音辨識優先使用：

```text
faster-whisper
```

不要直接使用：

```text
openai-whisper Python package
```

原因：

- faster-whisper 效能較好
- 記憶體使用較低
- CPU 執行較實用
- CUDA 支援較好
- 適合長時間錄音

核心套件：

```text
faster-whisper
ctranslate2
```

音訊處理使用：

```text
FFmpeg
```

Python CLI 可使用：

```text
Typer
Rich
```

如果不需要 Typer，也可使用 argparse，但 Rich 建議保留做進度顯示。

---

# 4. 專案目錄

建立以下結構：

```text
meeting-transcriber/
│
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
│
├─ run.bat
├─ run.ps1
│
├─ transcribe.py
│
├─ config.yaml
│
├─ src/
│  └─ meeting_transcriber/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ config.py
│     ├─ transcriber.py
│     ├─ audio.py
│     ├─ formatter.py
│     ├─ models.py
│     └─ utils.py
│
├─ input/
│
├─ output/
│  └─ .gitkeep
│
├─ models/
│  └─ .gitkeep
│
└─ tests/
   ├─ test_formatter.py
   ├─ test_config.py
   └─ test_utils.py
```

---

# 5. 最主要的使用方式

## 方式 A：直接拖錄音進 input

使用者將：

```text
新錄音.m4a
```

放到：

```text
input/
```

然後雙擊：

```text
run.bat
```

系統自動找出尚未處理的音訊檔並開始轉錄。

---

## 方式 B：命令列指定檔案

支援：

```powershell
python transcribe.py "C:\Recordings\新錄音.m4a"
```

或：

```powershell
python transcribe.py "C:\Recordings\新錄音.m4a" --model medium
```

或：

```powershell
python transcribe.py input
```

代表批次處理 input 目錄。

---

# 6. 支援格式

至少支援：

```text
.m4a
.mp3
.wav
.flac
.ogg
.mp4
.mov
.webm
```

FFmpeg 能解析的其他格式也可以接受。

---

# 7. 模型設定

預設：

```yaml
model: medium
language: zh
```

但允許修改：

```yaml
model: small
```

或：

```yaml
model: medium
```

或：

```yaml
model: large-v3
```

建議預設：

```text
medium
```

理由：

公司會議中會有：

- 中文
- 英文縮寫
- ERP
- Barcode
- Wafer
- Waffle
- iCube
- SMT
- PCB
- feeder
- 工令
- 批次
- 料號

需要比 tiny/base 更好的辨識能力。

---

# 8. GPU / CPU 自動偵測

程式啟動後自動判斷環境。

邏輯：

```python
if CUDA 可用:
    device = "cuda"
    compute_type = "float16"
else:
    device = "cpu"
    compute_type = "int8"
```

但不能只靠 torch 判斷。

需要確認 CTranslate2 是否真的能使用 CUDA。

如果 GPU 初始化失敗：

```text
CUDA initialization failed
```

必須：

1. 顯示警告
2. 自動 fallback CPU
3. 不要中止整個程式

例如：

```text
[WARN] CUDA 無法初始化，改用 CPU int8 模式。
```

---

# 9. 中文辨識

預設語言：

```text
zh
```

不要強制轉成簡體中文。

輸出盡可能保留繁體中文。

若 Whisper 結果包含簡體，可增加可選的繁簡轉換：

```yaml
traditional_chinese: true
```

可以使用：

```text
opencc-python-reimplemented
```

轉換：

```text
s2twp
```

但：

- 英文
- 型號
- Barcode
- API 名稱

不可被誤轉。

---

# 10. 專有名詞校正

建立：

```yaml
terminology:
  icube: iCube
  i cube: iCube
  iq cube: iCube
  wafer: Wafer
  waffle: Waffle
  bar code: Barcode
  barcode: Barcode
  pcb: PCB
  erp: ERP
  smt: SMT
```

另外加入：

```text
批次
批號
工令
工單
產品料號
材料料號
晶圓
晶片
載板
基板
槽位
上料
補料
追溯
物料防呆
```

但「專有名詞校正」只能做安全 replacement。

禁止因為 fuzzy matching 太強而修改正常語句。

例如：

```text
iQube → iCube
waffer → Wafer
```

可以。

不要對一般中文做猜測式替換。

---

# 11. Initial Prompt

呼叫 faster-whisper 時加入 initial prompt，改善專有名詞。

例如：

```text
本錄音為電子製造工廠 iCube 生產系統需求會議。
常見詞彙包括：
iCube、Wafer、Waffle、晶圓、晶片、載板、基板、
工令、工單、批次、批號、Barcode、ERP、
SMT、PCB、feeder、槽位、上料、補料、
物料防呆、生產追溯、產品料號、材料料號。
```

必須從：

```yaml
initial_prompt:
```

讀取。

不可寫死在 Python。

---

# 12. 長錄音處理

必須可處理：

```text
30 min
60 min
120 min
```

不能一次把整段音訊讀進 RAM。

優先使用 faster-whisper 原生 streaming / segment iterator。

例如：

```python
segments, info = model.transcribe(...)
```

逐段取得：

```python
for segment in segments:
```

每得到一段就寫入暫存結果。

不要等 1 小時全部辨識完才寫檔。

---

# 13. Checkpoint / Crash Recovery

這是必要功能。

長錄音如果跑到 50 分鐘時程式崩潰，不能全部重跑。

至少建立：

```text
output/<filename>/
    transcript.partial.jsonl
```

每完成一個 segment，立即追加：

```json
{
  "start": 123.4,
  "end": 127.8,
  "text": "這段是辨識結果"
}
```

程式異常結束時：

- 保留 partial
- 下次執行偵測 partial

如果無法真正從 Whisper 指定時間點 resume，可以：

1. 從最後 checkpoint 時間前 10～20 秒開始重新轉錄
2. 去除重複 segment

例如：

```text
checkpoint = 1835 sec
resume_from = 1820 sec
```

不要直接從 0 開始。

如果實作 resume 會過度複雜：

至少必須保證：

```text
partial transcript 不丟失
```

並在 README 註明限制。

---

# 14. 輸出目錄

每個錄音建立獨立目錄：

```text
output/
└─ 新錄音/
   ├─ 新錄音.txt
   ├─ 新錄音.timestamps.txt
   ├─ 新錄音.srt
   ├─ 新錄音.json
   ├─ 新錄音.md
   └─ metadata.json
```

---

# 15. TXT 格式

`新錄音.txt`

只保留容易貼給 AI 的內容。

例如：

```text
目前 Wafer 上有實體記號，但是我們還沒有廠商提供的 NG Map。

這部分要再確認。

接下來作業模式分成 Wafer 和 Waffle。

兩種模式都需要記錄材料 Barcode 與槽位 Barcode 的對應關係。
```

不要每句都帶 timestamp。

---

# 16. timestamps.txt

格式：

```text
[00:00:03 - 00:00:09]
目前 Wafer 上有實體記號。

[00:00:10 - 00:00:15]
但是我們目前沒有廠商提供的 NG Map。

[00:00:18 - 00:00:24]
這部分後續再確認。
```

時間格式：

```text
HH:MM:SS
```

---

# 17. SRT

產生標準：

```text
1
00:00:03,000 --> 00:00:09,100
目前 Wafer 上有實體記號。

2
00:00:10,000 --> 00:00:15,200
但是目前沒有廠商提供的 NG Map。
```

必須可正常被字幕軟體讀取。

---

# 18. JSON

格式：

```json
{
  "file": "新錄音.m4a",
  "duration": 3381.57,
  "language": "zh",
  "model": "medium",
  "device": "cuda",
  "compute_type": "float16",
  "created_at": "2026-09-17T09:00:00+08:00",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.1,
      "text": "目前 Wafer 上有實體記號。"
    }
  ]
}
```

---

# 19. Markdown 輸出

產生：

```text
新錄音.md
```

格式：

```markdown
# 新錄音－逐字稿

- 檔案：新錄音.m4a
- 長度：56:21
- 語言：中文
- 模型：medium

---

## 00:00–10:00

...

## 10:00–20:00

...

## 20:00–30:00

...
```

每 10 分鐘切一個 Markdown heading。

目的是方便：

- Claude Code
- Codex
- ChatGPT
- RAG

直接閱讀。

---

# 20. 進度顯示

執行時必須顯示：

```text
檔案：新錄音.m4a
長度：56:21
模型：medium
裝置：CUDA
語言：zh

Transcribing...

[████████████░░░░░░] 67%
37:48 / 56:21
```

如果 faster-whisper 無法取得精確百分比：

使用：

```text
目前 segment.end / audio_duration
```

計算。

---

# 21. 執行完成訊息

完成時：

```text
=================================
Transcription completed

File:
新錄音.m4a

Duration:
56:21

Processing:
18:42

Output:
output\新錄音\

Generated:
✓ TXT
✓ Timestamp TXT
✓ SRT
✓ JSON
✓ Markdown
=================================
```

---

# 22. Already Processed

避免同一錄音一直重跑。

可以使用：

```text
metadata.json
```

記錄：

```json
{
  "source_file": "新錄音.m4a",
  "size": 29014878,
  "mtime": 1780000000,
  "completed": true
}
```

如果：

```text
filename + size + mtime
```

皆一致：

顯示：

```text
[SKIP] 新錄音.m4a 已完成轉錄。
```

提供：

```text
--force
```

允許重新執行。

---

# 23. config.yaml

產生合理預設值：

```yaml
model: medium

language: zh

device: auto

compute_type: auto

traditional_chinese: true

beam_size: 5

vad_filter: true

output:
  txt: true
  timestamp_txt: true
  srt: true
  json: true
  markdown: true

input_dir: input
output_dir: output

initial_prompt: >
  本錄音為電子製造工廠 iCube 生產系統需求會議。
  常見詞彙包括 iCube、Wafer、Waffle、晶圓、晶片、
  載板、基板、工令、工單、批次、批號、Barcode、
  ERP、SMT、PCB、feeder、槽位、上料、補料、
  物料防呆、生產追溯、產品料號、材料料號。

terminology:
  iQube: iCube
  IQube: iCube
  icube: iCube
  "i cube": iCube
  waffer: Wafer
  wafer: Wafer
  waffle: Waffle
  barcode: Barcode
```

---

# 24. VAD

啟用：

```python
vad_filter=True
```

避免長時間：

- 沉默
- 工廠機器聲
- 空白

被 hallucination 成文字。

VAD 參數設計成 config 可調。

---

# 25. Whisper Hallucination 防護

工廠錄音可能有：

```text
機台聲
空調聲
無人講話
碰撞聲
```

Whisper 可能產生：

```text
謝謝觀看
字幕由 XXX 提供
請訂閱
```

需要增加基本異常片段過濾。

例如重複：

```text
謝謝觀看
感謝收看
請訂閱
字幕提供
```

若在無合理上下文大量出現，可標為 suspicious。

但是：

不要直接大量刪除正常內容。

JSON 可加入：

```json
{
  "text": "...",
  "suspicious": false
}
```

---

# 26. 不做自動摘要

這個工具的責任只有：

```text
Audio → Transcript
```

不要讓本地小模型自動生成會議摘要。

原因：

逐字稿稍後會交給：

```text
ChatGPT
Claude
Codex
```

整理。

因此：

```text
不要把原始逐字稿改寫成摘要。
```

---

# 27. 保留原始語意

可以做：

- 標點
- 繁簡轉換
- 明確專有名詞修正

不要做：

- 自動重寫
- 自動腦補
- 刪掉不確定的討論
- 自動把口語變成決議

原始 transcription 必須保留。

---

# 28. AI 後續使用格式

Markdown 最後加入：

```markdown
---

## AI 整理提示

請依據以上逐字稿整理正式會議記錄。

規則：

1. 不得自行新增逐字稿未提及的決議。
2. 將內容分類為：
   - 現況
   - 問題
   - 需求
   - 決議
   - 待確認事項
   - Action Items
3. 專有名詞：
   - iCube
   - Wafer
   - Waffle
   - Barcode
   - ERP
   - SMT
4. 使用「批次／批號」，不得寫成 P 次。
5. 如果內容無法確定，標示「待確認」，不要猜測。
```

這樣使用者可以：

```text
直接把整份 md 丟給 Agent
```

---

# 29. run.bat

提供可以直接雙擊的版本。

大致流程：

```bat
@echo off
cd /d %~dp0

if not exist .venv (
    echo Virtual environment not found.
    echo Please run setup first.
    pause
    exit /b 1
)

call .venv\Scripts\activate

python transcribe.py input

pause
```

---

# 30. setup.ps1

另外建立：

```text
setup.ps1
```

執行：

```powershell
.\setup.ps1
```

應完成：

1. 檢查 Python
2. 建立 `.venv`
3. pip upgrade
4. 安裝 requirements
5. 檢查 FFmpeg
6. 建立 input/output/models
7. 執行基本 self-test

不要要求系統管理員權限。

如果 FFmpeg 不存在：

請清楚顯示安裝方式。

如果能改用 Python 套件自動取得 ffmpeg，且穩定性足夠，也可以實作。

---

# 31. README

README 必須寫到完全沒碰過 Python 的人也能用。

至少包含：

## 安裝

```powershell
git clone ...
cd meeting-transcriber
powershell -ExecutionPolicy Bypass -File setup.ps1
```

## 使用

把：

```text
錄音.m4a
```

放入：

```text
input
```

雙擊：

```text
run.bat
```

## 指定檔案

```powershell
python transcribe.py "C:\xxx\meeting.m4a"
```

## 改模型

```yaml
model: small
```

## CPU 建議

```text
small
medium
```

## GPU 建議

```text
medium
large-v3
```

---

# 32. 錯誤處理

至少處理：

### File not found

```text
[ERROR] 找不到輸入檔案
```

### Unsupported audio

```text
[ERROR] 無法解析音訊格式
```

### FFmpeg missing

```text
[ERROR] FFmpeg 尚未安裝
```

### Model download failed

```text
[ERROR] Whisper 模型下載失敗
```

### CUDA unavailable

不可當 ERROR。

顯示：

```text
[WARN] CUDA 不可用，使用 CPU。
```

然後繼續。

---

# 33. 模型下載

第一次執行模型可自動下載。

模型使用 Hugging Face / faster-whisper 預設下載機制。

下載完成後必須可以：

```text
離線執行
```

不要每次重新下載。

README 註明：

```text
第一次需要網路下載模型。
之後轉錄可以完全離線。
```

---

# 34. Log

建立：

```text
logs/
```

例如：

```text
logs/transcribe-20260917.log
```

記錄：

```text
開始時間
來源檔案
模型
device
compute_type
audio duration
segment 數
處理時間
exception traceback
```

Console 不需要顯示完整 traceback。

Log 才顯示。

---

# 35. 隱私

工具首頁 / README 明確註明：

```text
所有錄音與逐字稿均於本機處理。
程式不會主動將音訊上傳到任何第三方服務。
```

除模型第一次下載外，不應有網路需求。

---

# 36. 單元測試

至少測試：

```text
timestamp formatting
SRT formatting
terminology normalization
config loading
output path
already processed detection
Traditional Chinese conversion
```

不需要 CI 才能完成。

執行：

```powershell
pytest
```

應全部通過。

---

# 37. 程式品質

要求：

- type hints
- pathlib
- dataclass / Pydantic model 擇一
- functions 不要全部塞在 transcribe.py
- 不使用全域 mutable state
- exception 有明確分類
- Windows path 正確處理
- 中文檔名正常
- UTF-8 輸出
- 不使用 shell=True 執行外部指令

---

# 38. 不可接受的實作

以下都視為未完成：

```text
只支援 wav
```

```text
要我先人工轉 m4a
```

```text
只能轉前 60 秒
```

```text
使用 OpenAI API
```

```text
要求 AccurateScribe
```

```text
沒有 CPU fallback
```

```text
沒有產生 TXT
```

```text
一小時錄音必須整段完成後才會寫結果
```

```text
程式出錯後什麼都沒留下
```

---

# 39. 驗收情境

使用這個情境驗收：

來源：

```text
新錄音.m4a
```

條件：

```text
duration ≈ 56 minutes
language = Chinese
contains English technical terms
```

執行：

```powershell
python transcribe.py "新錄音.m4a"
```

預期：

```text
✓ 能完整讀取 56 分鐘錄音
✓ 顯示辨識進度
✓ 不只輸出前 60 秒
✓ 產生完整 TXT
✓ 產生 timestamps TXT
✓ 產生 SRT
✓ 產生 JSON
✓ 產生 Markdown
✓ iCube 不被錯寫成 iQube
✓ Wafer / Waffle 盡可能正確
✓ 批次相關中文可正常顯示
✓ UTF-8 無亂碼
```

---

# 40. 最終 Agent 任務

請不要只建立 prototype。

請直接：

1. 建立完整專案。
2. 實作所有必要程式。
3. 建立虛擬環境設定。
4. 建立 requirements。
5. 建立 config。
6. 建立 Windows `setup.ps1`。
7. 建立 `run.bat`。
8. 建立 README。
9. 建立測試。
10. 執行測試。
11. 修正所有錯誤。
12. 實際執行 CLI smoke test。
13. 確認 Windows path / 中文檔名沒有問題。
14. 不要留 TODO 取代必要功能。

如果目前 repo 中已有程式：

- 先閱讀現有程式
- 儘量整合
- 不要無理由大改架構

完成後請輸出：

```text
IMPLEMENTATION SUMMARY

完成項目：
...

測試：
...

執行方式：
...

已知限制：
...
```

如果遇到非必要的小問題，請自行做合理決策，不要停下來詢問使用者。

目標是讓使用者完成後可以直接：

```text
1. 把 m4a 放到 input
2. 雙擊 run.bat
3. 等待完成
4. 打開 output/錄音名稱/錄音名稱.md
5. 把 md 交給 AI 整理會議紀錄
```

這才算完成。