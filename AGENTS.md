# 本地多模态文件语义索引与搜索 — SPEC（适用于 gemini-cli / codex）

> 目标：在 **完全离线 / 本地** 环境中，为本地文件建立索引，支持用“语义 + 模糊需求”检索相关文件。覆盖常见多媒体格式与压缩包；索引后台运行；系统负荷高时自动暂停（带 debounce 防抖）；使用小模型（CLIP、Ollama gemma3:1b、Whisper）；运行在 Apple Silicon M1 + PyTorch（优先 MPS）。

---

## 1. 项目概述

### 1.1 项目名称
`local-semantic-indexer`（简称 `lsi`）

### 1.2 核心能力
- 本地文件扫描与增量索引（支持后台常驻）
- 多模态内容抽取与向量化（文本 / PDF / 图片 / 音频 / 视频 / 压缩包）
- 混合检索：**语义向量检索 + 模糊/关键词检索（文件名/文本） + 元数据过滤**
- 本地 LLM（Ollama gemma3:1b）用于：
  - 查询理解与扩展（将“模糊需求”转成可检索条件）
  - 结果 rerank（可选，默认开启但可配置）
  - 生成解释（为什么命中）
- Whisper 用于音频/视频语音转写
- CLIP 用于跨模态对齐（文本查询可搜图片/视频帧；文本也能统一编码）

---

## 2. 明确约束（硬性要求）

### 2.1 完全本地
- **禁止任何外网请求**（包括模型下载、遥测、在线 API）。
- 允许访问本机 `ollama`（localhost）作为本地推理服务。
- 依赖安装需要用户提前完成（如 `ffmpeg`、Python 包等），运行期不自动下载模型。

### 2.2 规避敏感文件
- 默认跳过：
  - `~/.ssh/**`
  - `~/.gnupg/**`
  - `~/Library/Keychains/**`
  - `**/.env`、`**/.pem`、`**/*.key`、`**/*id_rsa*` 等（可配置）
  - 系统目录：`/System/**`、`/Library/**`（可配置）
- 必须支持用户通过配置文件追加 `ignore_patterns` 与 `allow_roots`（仅在允许根目录内工作）。

### 2.3 压缩包策略
- **> 1GB** 的压缩包：只 **peek**（读取内部文件名列表与基础元数据），**不真正解压**。
- **≤ 1GB** 的压缩包：允许临时解压到缓存目录，按文件类型正常处理；完成后清理。
- 压缩包内文件的“虚拟路径”必须可定位，例如：
  - `~/Downloads/a.zip::docs/readme.txt`

### 2.4 后台索引 + 系统负荷自适应暂停（debounce）
- 索引任务在后台队列执行，可常驻。
- 当系统负荷高（CPU、LoadAvg、可用内存、磁盘 I/O 任一指标超过阈值）时，自动暂停队列消费。
- 必须带 debounce 防抖：
  - 连续超过阈值 `pause_after_s` 才暂停
  - 连续低于阈值 `resume_after_s` 才恢复
- 暂停时不丢任务；恢复后继续。

### 2.5 运行环境
- Apple Silicon M1
- PyTorch（优先使用 `mps` device；不可用则回退 CPU）
- Python（建议 3.11+）

---

## 3. 范围与非目标

### 3.1 In Scope（必须实现）
- 递归扫描目录、过滤敏感路径
- 文件内容抽取（按格式）
- 多模态向量化
- 向量索引与查询
- 关键词/模糊检索（至少文件名 + 文本内容）
- 混合排序与去重（同一文件多个 chunk/帧/段聚合）
- 后台索引守护进程（至少一个 `--daemon` 模式）
- 自动暂停/恢复（debounce）
- 可配置（阈值、根目录、忽略规则、压缩包阈值、模型开关等）
- 可观测性：日志、进度、索引状态查看

### 3.2 Out of Scope（明确不做，或可后续）
- 云端同步、多人共享索引
- 对加密压缩包/加密 PDF 的破解（遇到就跳过并记录）
- 高精度 OCR（可留接口，但默认不启用）
- GUI（仅 CLI + 可选本地 HTTP API）

---

## 4. 支持的文件类型与处理策略

### 4.1 文本类
- 扩展名：`.txt .md .rst .json .yaml .yml .csv .log .py .js .ts .go .java ...`
- 抽取：直接读取（按编码容错，优先 UTF-8，失败则用 `chardet` 或忽略错误）
- 分块：按字符/句子/段落 chunk（目标每块 ~400–800 tokens 等价长度）
- 向量：CLIP Text Encoder（统一跨模态）
- 额外：存储 chunk 的短 snippet（用于展示命中片段）

### 4.2 PDF
- 抽取优先级：
  1) 可提取文本：`pypdf` / `pdfminer.six` 读取文本
  2) 扫描型 PDF：默认仅记录“不可提取文本”，可选启用 OCR（接口预留）
- 分块：按页或按段落；每页可拆成多个 chunk
- 向量：CLIP Text Encoder（对文本）

### 4.3 图片（jpg/png/webp 等）
- 抽取：读取图片、缩放至模型输入尺寸
- 向量：CLIP Image Encoder
- 元数据：宽高、EXIF（可选，注意隐私，默认只保存必要字段）

### 4.4 音频（wav/mp3/m4a 等）
- 抽取：Whisper 转写（tiny/base，默认 tiny 优先速度）
- 分块：按时间段或按句子（例如每 20–40 秒一个 chunk，并附带时间戳）
- 向量：对转写文本使用 CLIP Text Encoder
- 元数据：时长、采样率

### 4.5 视频（mp4/mov 等）
- 依赖：本地 `ffmpeg`
- 抽取：
  - 音轨：ffmpeg 提取音频 -> Whisper 转写
  - 画面：抽关键帧（每 N 秒一帧，默认 2s 或可配置；或简单场景变化检测可后续）
- 向量：
  - 帧：CLIP Image Encoder
  - 转写：CLIP Text Encoder
- 聚合：同一视频的帧/转写命中合并成一个结果，展示最佳命中片段/时间戳

### 4.6 压缩包（zip/tar/7z 等）
- 支持格式（优先实现）：
  - `.zip`（Python `zipfile`）
  - `.tar/.tar.gz/.tgz`（`tarfile`）
  - `.7z`（`py7zr`，若不可用则仅记录“未支持”）
- 行为：
  - 大包（> 1GB）：只 peek 文件名列表 + 大小（若可得）+ 修改时间（若可得），并索引“文件名文本”（用于搜索包内可能有什么）
  - 小包（≤ 1GB）：解压到缓存 `cache/extract/<hash>/...`，逐文件按其类型处理；完成后清理目录
- 安全：解压必须防止 ZipSlip（路径穿越），必须校验解压目标路径在缓存目录内

---

## 5. 总体架构

### 5.1 模块图（逻辑）
```

CLI / API
|
v
Query Understanding (Gemma via Ollama, optional)
|
+--> Text embedding (CLIP text)
|
v
Hybrid Retrieval
|-- Vector search (HNSW/FAISS) over chunks/frames/transcripts
|-- Lexical search (SQLite FTS5) over file names + text snippets
|-- Metadata filter (path/type/time/size)
v
Aggregation + Rerank (Gemma optional)
v
Results (paths + snippets + reason)

```

### 5.2 索引流水线
```

Scanner -> Filter -> Extract -> Chunk -> Embed -> Persist (DB + Vector Index)
|
+-> Temp extract (archives <= 1GB) -> cleanup

````

---

## 6. 数据存储设计

### 6.1 存储位置（macOS 建议）
- 索引根目录：`~/Library/Application Support/local-semantic-indexer/`
  - `index.sqlite3`（元数据 + FTS）
  - `vectors/`（向量索引文件）
  - `cache/`（临时解压、视频帧缓存，可定期清理）
  - `logs/`

### 6.2 SQLite 表（建议）
#### 6.2.1 files
- `file_id` (PK)
- `path`（真实路径；压缩包内用虚拟路径）
- `real_path`（真实文件路径；对压缩包内文件，real_path=压缩包路径）
- `type`（text/pdf/image/audio/video/archive）
- `mime`（可选）
- `size_bytes`
- `mtime_ns`
- `ctime_ns`（可选）
- `hash`（可选：内容 hash 或快速 hash）
- `status`（indexed / skipped / error）
- `error_message`（可选）
- `created_at`, `updated_at`

#### 6.2.2 chunks（文本/PDF/转写）
- `chunk_id` (PK)
- `file_id` (FK)
- `chunk_type`（text/pdf_page/audio_segment/video_transcript）
- `start_offset` / `end_offset`（字符偏移或页码）
- `start_time_ms` / `end_time_ms`（音频/视频）
- `snippet`（用于展示，长度限制如 512 chars）
- `vector_id`（指向向量索引条目）
- `created_at`

#### 6.2.3 frames（图片/视频帧）
- `frame_id` (PK)
- `file_id` (FK)
- `timestamp_ms`（视频帧；图片可为空）
- `width`, `height`
- `vector_id`
- `created_at`

#### 6.2.4 archive_entries（大压缩包 peek）
- `entry_id` (PK)
- `file_id` (FK; 指向 archive 本体)
- `inner_path`
- `size_bytes`（若可得）
- `mtime_ns`（若可得）

#### 6.2.5 FTS
- `fts_files`：至少包含
  - `path`
  - `basename`
  - `snippet`（聚合一些 chunk snippet 或首段）
  - `archive_inner_paths`（大包内文件名串）
- 实现：SQLite FTS5 虚表 + contentless 或外部内容表皆可

### 6.3 向量索引
- 推荐先实现：`hnswlib`（纯本地、轻量、易用）
- 备选：FAISS（在 M1 上安装可能更复杂）
- 向量空间：
  - 统一 CLIP embedding 维度（取决于选用的 CLIP 模型）
- 分桶（可选但建议）：
  - `clip_text_index`
  - `clip_image_index`
  - 查询时用文本 embedding 同时查 text_index 与 image_index（跨模态）

---

## 7. 模型与推理

### 7.1 CLIP
- 通过 `open_clip`（PyTorch）加载本地权重
- Device：优先 `mps`，否则 CPU
- 批处理：图片/文本 embedding 都要 batch，提高吞吐
- 图片预处理：缩放、中心裁剪、标准化（遵循 open_clip 推荐）

### 7.2 Whisper
- 使用 PyTorch 版本（`openai-whisper`）
- 模型：`tiny` 默认；可配置为 `base`
- 运行策略：
  - 音频较长时分段处理
  - 保存时间戳（用于搜索结果定位）
- Device：尽量使用 `mps`（若 whisper 对 mps 不稳定则回退 CPU，并记录日志）

### 7.3 Ollama gemma3:1b
- 通过本地 HTTP 调用 `ollama`（仅 `http://127.0.0.1`）
- 用途（可配置开关）：
  1) `query_parse`: 将用户自然语言变成结构化 query（关键词、类型偏好、过滤条件）
  2) `rerank`: 对候选结果进行重排（输入 topK 的标题/片段）
  3) `explain`: 给出命中解释（短句）
- 必须提供 `--no-llm` 模式：只做 embedding + FTS，不调用 gemma

---

## 8. 后台索引与负荷自适应暂停

### 8.1 工作队列
- 采用 `asyncio` + 有界队列（或线程池）
- 并发策略：
  - I/O 密集（扫描/读取）可多线程
  - 模型推理（CLIP/Whisper）并发要限制（默认 1–2）
- 支持任务优先级：
  - 文件变更（增量）优先于全量重扫

### 8.2 系统负荷监控
- 依赖：`psutil`
- 指标（默认建议）：
  - CPU：`cpu_percent`（例如 > 70% 视为高负荷）
  - LoadAvg：`os.getloadavg()`（例如 1min load > 核心数 * 0.8）
  - 内存：available < 某阈值（例如 < 2GB）
  - 磁盘：可选（简单用 I/O 等待或队列长度指标替代）
- Debounce 参数（可配置）：
  - `pause_after_s`: 10s
  - `resume_after_s`: 30s
- 行为：
  - 超阈值持续 `pause_after_s`：暂停消费队列（worker 进入 idle）
  - 低于阈值持续 `resume_after_s`：恢复 worker
- 必须提供 `lsi status` 显示当前暂停原因与持续时间

---

## 9. CLI 设计（必须）

### 9.1 基础命令
- `lsi init`
  - 初始化数据目录、创建 SQLite、创建向量索引文件结构、写默认 config

- `lsi index --roots <path...> [--daemon] [--once]`
  - `--daemon`: 常驻后台（建议写 PID 文件 + 日志）
  - `--once`: 扫描一次退出
  - `--rescan`: 忽略增量状态，强制重建（或重算 embedding）
  - `--dry-run`: 仅显示将处理哪些文件，不落库

- `lsi search "<query>" [--top 20] [--type image|text|pdf|audio|video|archive] [--path <glob>] [--since <date>]`
  - 返回：文件路径、命中类型、片段/时间戳、综合分数、解释（若启用 LLM）

- `lsi status`
  - 显示：索引进度、队列长度、最近错误、是否暂停（及原因）、索引库大小

- `lsi config show|set`
  - 读取/修改配置（或直接编辑 YAML/JSON）

- `lsi cleanup`
  - 清理临时缓存（解压目录、视频帧缓存）

### 9.2 输出格式
- 默认：人类可读
- 可选：`--json` 输出（便于脚本/集成）

---

## 10. 配置文件（示例）

保存为：
`~/Library/Application Support/local-semantic-indexer/config.yaml`

```yaml
roots:
  - "/Users/<you>/Documents"
  - "/Users/<you>/Downloads"

ignore_patterns:
  - "**/.ssh/**"
  - "**/.gnupg/**"
  - "**/Library/Keychains/**"
  - "**/.env"
  - "**/*.pem"
  - "**/*.key"
  - "**/*id_rsa*"

archive:
  large_threshold_bytes: 1073741824   # 1GB
  enable_extract_small: true
  extract_cache_max_gb: 10

models:
  clip:
    name: "ViT-B-32"          # 具体由 open_clip 支持
    device_prefer: "mps"
    batch_size: 32
  whisper:
    model_size: "tiny"
    device_prefer: "mps"
  ollama:
    enabled: true
    base_url: "http://127.0.0.1:11434"
    model: "gemma3:1b"

indexing:
  max_workers_io: 4
  max_workers_model: 1
  chunk_chars: 2000
  chunk_overlap_chars: 200

load_shedding:
  cpu_pause_percent: 70
  mem_available_min_gb: 2
  pause_after_s: 10
  resume_after_s: 30

video:
  frame_every_seconds: 2.0
  max_frames_per_video: 2000

security:
  local_only: true
  disallow_network: true
````

---

## 11. 检索与排序（混合策略）

### 11.1 召回（Recall）

* 语义召回：

  * `q_embed = CLIP.text(query)`
  * 在 text chunks 与 image/video frames 索引中各取 topK（如各 50）
* 词法召回：

  * SQLite FTS5 对 `path/basename/snippet/archive_inner_paths` 检索 topK（如 50）
* 合并去重：按 `file_id` 聚合

  * 聚合时保存：

    * 最佳 chunk/snippet 或最佳 frame timestamp
    * 命中来源（vector/fts）

### 11.2 排序（Rank）

* 初排：加权融合分数

  * `score = w_vec * sim + w_fts * bm25 + w_meta * meta_bonus`
* 可选 rerank：

  * 用 gemma3:1b 对 topN（如 30）做重排
  * 输入：query + 每个候选的（文件名、路径、snippet、时间戳）
  * 输出：排序 + 每条 1 句解释（限制长度）

---

## 12. 增量更新策略

* 每次扫描记录 `mtime_ns` + `size_bytes` +（可选）快速 hash
* 若文件未变化：跳过重新抽取/embedding
* 删除：可通过周期性 “墓碑扫描” 标记 `status=deleted` 或清理记录
* 目录监听（可选增强）：

  * macOS 可用 `watchdog`（FSEvents）监听 roots，实时入队变更

---

## 13. 错误处理与可恢复性

* 任意文件处理失败（权限/损坏/格式不支持/解压失败）：

  * `files.status=error` + 记录 `error_message`
  * 不影响整体索引继续
* 临时解压目录：

  * 使用内容 hash 命名，支持崩溃后 `cleanup` 清理
* 处理超大文件/视频：

  * 限制最大帧数、最大处理时长（可配置），超出则截断并记录

---

## 14. 依赖与系统要求

### 14.1 Python 依赖（建议）

* `torch`（MPS）
* `open_clip_torch`
* `whisper`（openai-whisper）
* `psutil`
* `pypdf` 或 `pdfminer.six`
* `Pillow`
* `hnswlib`
* `sqlite-utils`（可选）或直接 `sqlite3`
* `watchdog`（可选）

### 14.2 系统依赖

* `ffmpeg`（视频/音频抽取）
* `ollama`（本地运行 gemma3:1b）

---

## 15. 仓库结构（建议）

```
local-semantic-indexer/
  SPEC.md
  pyproject.toml
  lsi/
    __init__.py
    cli.py
    config.py
    paths.py
    scanner.py
    filters.py
    extractors/
      text_extractor.py
      pdf_extractor.py
      image_extractor.py
      audio_extractor.py
      video_extractor.py
      archive_extractor.py
    chunking.py
    embedding/
      clip_embedder.py
      whisper_transcriber.py
      ollama_client.py
    storage/
      db.py
      fts.py
      vectors.py
    scheduler/
      queue.py
      load_monitor.py
      daemon.py
    search/
      hybrid.py
      rerank.py
      aggregation.py
  tests/
    test_filters.py
    test_archives.py
    test_chunking.py
    test_load_monitor.py
    test_search_smoke.py
```

---

## 16. 验收标准（Definition of Done）

### 16.1 功能验收

* [ ] 能对指定 roots 建立索引并持久化（重启后可用）
* [ ] 支持搜索：文本 query 能命中文本/PDF、图片、音频转写、视频帧/转写
* [ ] 支持压缩包：

  * [ ] >1GB：能检索到包内文件名（peek）且不解压
  * [ ] ≤1GB：能临时解压索引并清理
* [ ] 默认能避开 `.ssh` 等敏感路径（可配置）
* [ ] 后台索引运行时，系统负荷高会自动暂停，负荷降低后自动恢复（debounce 生效）
* [ ] `lsi status` 能看到暂停/恢复状态与原因
* [ ] 全程无外网请求（除了 localhost 的 ollama）

### 16.2 性能/资源验收（基线）

* [ ] 在 M1 上 CLIP embedding 可用（MPS 或 CPU fallback）
* [ ] Whisper 转写可用（MPS 不可用则 CPU fallback）
* [ ] 大量文件下不会阻塞交互式 search（index 与 search 资源隔离/限流）

---

## 17. 交付物

* 可运行的 CLI：`lsi`
* 本地索引目录与配置文件
* README（包含安装步骤、如何安装 ffmpeg/ollama、示例命令）
* 测试用例（至少覆盖过滤、压缩包策略、负荷暂停、防 ZipSlip）

---

## 18. 示例使用流程

```bash
# 初始化
lsi init

# 启动后台索引
lsi index --roots ~/Documents ~/Downloads --daemon

# 查看状态
lsi status

# 搜索（模糊需求）
lsi search "上周跟客户会议的录音，提到了报价" --top 20

# 搜索图片/视频相关（文本搜图）
lsi search "白板上写着 roadmap 的照片" --top 20

# 输出 JSON 便于脚本处理
lsi search "2023 Q4 报告 pdf" --json

# 清理临时缓存
lsi cleanup
```

