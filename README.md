# local-semantic-indexer (lsi)

在 **完全本地/离线** 环境中，对本地文件建立多模态语义索引，支持用自然语言描述进行跨模态检索。覆盖文本、PDF、图片、音频、视频、压缩包等常见格式。

## 核心特性

- **完全离线**：禁止任何外网请求（仅允许 `127.0.0.1/localhost` 用于 Ollama，可配置开关）
- **多模态语义搜索**
  - 文本搜图片：用文字描述找到相关图片/视频帧
  - 文本搜视频：找到视频中的关键帧和语音转写片段
  - 语义理解：支持自然语言描述，不只是关键词匹配
- **双向量引擎**
  - **CLIP** (`ViT-B-32`)：图像-文本跨模态对齐
  - **Sentence-Transformers** (`all-MiniLM-L6-v2`)：更好的中文文本语义理解
- **语音转写**：使用 Whisper 将音频/视频中的语音转为可搜索的文本
- **压缩包智能处理**
  - `> 1GB`：仅读取内部文件名列表（不解压）
  - `≤ 1GB`：临时解压索引后清理；虚拟路径：`/path/a.zip::inner/file.txt`
  - 自动防护 ZipSlip 攻击
- **Apple Silicon 优化**：优先使用 MPS 加速（CLIP、Whisper、sentence-transformers）
- **后台守护进程**：`--daemon` 模式支持持续监控
- **系统负荷自适应**：CPU/内存/LoadAvg 超阈值时自动暂停索引，恢复后自动继续
- **可选本地 LLM**（Ollama）：查询理解、结果重排序、生成解释

## 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone <repo-url>
cd local-semantic-indexer

# 安装（开发模式）
pip install -e '.[pdf,vectors,clip,whisper,archives]'

# 如需语音转写，下载 Whisper 模型到本地缓存
# 模型会缓存到 ~/.cache/whisper/

# 需要 ffmpeg（用于视频/音频处理）
# macOS: brew install ffmpeg
```

### 2. 初始化

```bash
# 创建数据目录和默认配置
lsi init
```

数据目录默认位置：`~/Library/Application Support/local-semantic-indexer/`

目录结构：
```
~/Library/Application Support/local-semantic-indexer/
├── config.yaml          # 配置文件
├── index.sqlite3        # 元数据 + FTS5 全文索引
├── vectors/             # 向量索引文件
│   ├── clip_image.hnsw      # CLIP 图像向量 (512维)
│   ├── clip_text.hnsw       # CLIP 文本向量 (512维)
│   └── st_text.hnsw         # Sentence-Transformers 文本向量 (384维)
├── cache/               # 临时缓存（解压、视频帧等）
└── logs/                # 日志文件
```

**提示**：可用 `LSI_APP_DIR` 环境变量覆盖数据目录。

### 3. 建立索引

```bash
# 扫描一次并退出（默认索引 ~/Desktop）
lsi index --once

# 指定目录
lsi index --roots ~/Documents ~/Downloads --once

# 后台常驻索引（持续监控文件变化）
lsi index --daemon

# 强制重建索引
lsi index --once --rescan

# 仅预览要索引的文件（不入库）
lsi index --once --dry-run
```

### 4. 语义搜索

```bash
# 基础搜索
lsi search "女王" --top 10

# 按类型过滤
lsi search "截图" --type image
lsi search "会议录音" --type video

# JSON 输出（便于脚本处理）
lsi search "项目报告" --json

# 禁用 LLM（纯向量+FTS 检索）
lsi search "文档" --no-llm
```

### 5. 查看状态

```bash
lsi status
# 输出示例：
# Queue: 0 | Processed: 0 | Skipped: 0 | Errors: 413 | DB: 1.6MB | Vectors: 4.5MB
# Paused: no
```

## 配置说明

配置文件：`~/Library/Application Support/local-semantic-indexer/config.yaml`

### 常用配置项

```yaml
# 索引范围
roots:
  - "/Users/<username>/Desktop"
  - "/Users/<username>/Downloads"

# 安全护栏：只允许在这些目录内工作
allow_roots:
  - "/Users/<username>/Desktop"
  - "/Users/<username>/Downloads"

# 忽略规则
ignore_patterns:
  - "**/.ssh/**"
  - "**/.env"
  - "**/*.pem"
  - "**/.git/**"
  - "**/node_modules/**"

# CLIP 配置（图像-文本跨模态）
models:
  clip:
    name: "ViT-B-32"
    pretrained: "laion2b_s34b_b79k"  # 或 "openai"
    backend: "open_clip"
    device_prefer: "mps"              # mps | cpu
    batch_size: 32

  # Sentence-Transformers 配置（纯文本语义）
  text_embedder:
    backend: "sentence_transformers"
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    device_prefer: "mps"
    batch_size: 32

  # Whisper 语音转写
  whisper:
    model_size: "base"    # tiny | base | small
    device_prefer: "mps"
    enabled: true

  # Ollama 本地 LLM（可选）
  ollama:
    enabled: true
    base_url: "http://127.0.0.1:11434"
    model: "gemma3:1b"

# 索引参数
indexing:
  max_workers_io: 4
  max_workers_model: 1
  chunk_chars: 2000
  chunk_overlap_chars: 200

# 视频处理
video:
  frame_every_seconds: 2.0    # 每2秒提取一帧
  max_frames_per_video: 2000

# 系统负荷控制
load_shedding:
  cpu_pause_percent: 70
  mem_available_min_gb: 2
  pause_after_s: 10
  resume_after_s: 30
```

## 搜索示例

### 1. 中文语义搜索

数据库中的视频转写片段：
```
/Users/georgezhou/Desktop/short.mov @45000ms: "就是奋怒女王"
```

搜索：
```bash
$ lsi search "女王" --top 5
0.586 [video] ~/Desktop/short.mov @45000ms (vector:image,vector:text)
  就是奋怒女王
```

### 2. 文本搜图片

```bash
$ lsi search "截图" --type image --top 5
0.224 [image] ~/Desktop/Screenshot 2025-07-23 at 21.25.57.png (vector:image)
0.223 [image] ~/Desktop/Screenshot 2025-07-24 at 17.07.52.png (vector:image)
```

### 3. 语义相似匹配

```bash
# 搜索语义相似的内容（不需要关键词完全匹配）
$ lsi search "我喜欢这个开场" --top 3
0.683 [video] ~/Desktop/short.mov @8000ms (vector:image,vector:text)
  对啊 我爱这双巴老师
```

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                      CLI / API                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  查询理解 (Ollama, 可选)                                 │
│  - 将自然语言转为结构化查询                               │
└─────────────────────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
   ┌──────────────────┐    ┌──────────────────┐
   │  文本嵌入器       │    │   CLIP 文本编码   │
   │  (Sentence-TF)   │    │   (跨模态检索)    │
   └────────┬─────────┘    └────────┬─────────┘
            │                       │
            ▼                       ▼
   ┌──────────────────┐    ┌──────────────────┐
   │  ST_Text Index   │    │  CLIP_Text Index │
   │   (384 dim)      │    │   (512 dim)      │
   └────────┬─────────┘    └────────┬─────────┘
            │                       │
            └───────────┬───────────┘
                        ▼
        ┌───────────────────────────────┐
        │      Hybrid Retrieval         │
        │  - 语义召回（向量搜索）        │
        │  - 词法召回（SQLite FTS5）     │
        │  - 元数据过滤（路径/时间/类型） │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │   Rerank / Explain (Ollama)   │
        └───────────────┬───────────────┘
                        ▼
                   搜索结果
```

## 模型说明

| 模型 | 用途 | 维度 | 离线可用 |
|------|------|------|----------|
| CLIP (ViT-B-32) | 图像-文本跨模态 | 512 | ✅ |
| all-MiniLM-L6-v2 | 文本语义理解 | 384 | ✅ |
| Whisper (base) | 语音转写 | - | ✅ |
| gemma3:1b (Ollama) | 查询理解/重排序 | - | ✅ |

**注意**：首次使用时会自动从 HuggingFace 缓存加载模型（需要预先下载或使用已缓存的模型）。

## 系统要求

- **操作系统**：macOS（已测试），Linux 应该也支持
- **Python**：3.10+
- **硬件**：Apple Silicon (M1/M2/M3) 推荐，支持 MPS 加速
- **依赖**：ffmpeg（视频/音频处理）

## 开发

```bash
# 运行测试
python -m unittest discover -s tests -p 'test_*.py'

# 调试模式
python -m lsi search "测试" --top 5
```

## 注意事项

1. **隐私保护**：默认跳过敏感路径（.ssh、Keychains、.env 等）
2. **安全护栏**：只允许在 `allow_roots` 配置的目录内工作
3. **网络限制**：默认禁止外网请求，仅允许本地 Ollama
4. **模型缓存**：运行期不会自动下载模型，需要预先准备或使用缓存

## 许可证

MIT
