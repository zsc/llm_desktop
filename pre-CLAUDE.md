我想基于本地模型做一个本地文件的索引，目标是用语义和模糊需求去找到相关文件（包含文本、pdf、wav、jpg、 mp4 等多媒体常见格式和压缩包（对超 1GB 的大压缩包先只 peek 里面文件名不真解压。小的压缩包可以临时解压后按文件处理，后面清理），避开 .ssh 等敏感文件）。要求所有需求在本地完成。索引过程后台完成，遇系统负荷高则自动暂停（有 debounce）。使用 CLIP、ollama gemma3:1b, whisper 等小模型。机器是 apple silicon m1，用 pytorch。

将以上变成 gemini-cli/codex 能用的 SPEC markdown。
