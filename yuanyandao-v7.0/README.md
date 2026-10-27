# 元衍道枢 YuanYanDaoShu V7.0

> 私有 RAG 沙盘推演调度插件 ｜ 适配 DeepSeek Harness ｜ RTX5060 8G ｜ Qwen2.5-7B-Base 4bit QLoRA
> **核心锚点密钥：元衍道枢**（不可修改、不可删除、不可绕过）

元衍道枢不是通用大模型，而是**调度路由引擎 + 检索分发器 + 沙盘推演控制器 + 训练任务调度器**。
它接收上层「元衍智枢」或用户输入，完成双路检索、五级权限校验、意图分流、输出校验与审计留痕，
最后返回结构化结果。

| 角色 | 定义 | 边界 |
| --- | --- | --- |
| 元衍道枢 | 底层调度内核插件（本仓库） | 不做终端交互、不做开放域闲聊 |
| 元衍智枢 | 面向终端的业务中台产品 | 不实现检索、鉴权、审计内核 |

---

## 1. 交付物清单

| # | 要求产物 | 位置 |
| --- | --- | --- |
| 1 | 插件主程序 Python 源码 | `yuanyandao/`（`dispatcher.py` 为主循环） |
| 2 | 向量库接入配置 YAML（FAISS / Chroma） | `config/config.yaml`、`config/vector_store.faiss.yaml`、`config/vector_store.chroma.yaml` |
| 3 | 基座 System Prompt | `system_prompt.md` |
| 4 | 关键词批量入库脚本 | `scripts/import_keywords.py` |
| 5 | 沙盘推演并行分支 + 风险打分 | `yuanyandao/sandbox.py` |
| 6 | 五级权限鉴权 + 动态密钥 | `yuanyandao/permissions.py` |
| 7 | JSONL 审计 + 快照归档 | `yuanyandao/audit.py` |
| 8 | 部署文档 / 一键启动 / Ollama 示例 | 本 README、`run_rtx5060.ps1`、`run_rtx5060.bat` |
| 9 | 全套测试用例 JSON | `tests/test_cases_v7.json`、`tests/run_tests.py` |

附加：`yuanyandao/server.py`（HTTP 服务）、`yuanyandao/mcp_server.py`（MCP stdio 接入）、
`yuanyandao/documents.py`（BP / 交付规范模板）。

---

## 2. 架构与主循环

```text
用户 / 元衍智枢
      │  invoke(text, user_id, level, api_key)
      ▼
┌──────────────────────────────────────────────────────┐
│ 元衍道枢调度内核                                       │
│ STEP1 关键词精确提取 + 语义向量编码 → 双路匹配          │
│ STEP2 五级权限校验（检索之前）→ 不匹配即拒绝并留痕       │
│ STEP3 按知识库分类路由，召回 Top-N，附来源标签          │
│ STEP4 意图分流：问答 / 沙盘 / 文档 / 训练 / 词库维护     │
│ STEP5 输出校验：锚点、角色边界、幻觉、违禁内容           │
│ STEP6 封装结构化结果 + 调用元数据 + 审计与快照           │
│ STEP7 回传上层入口                                     │
└──────────────────────────────────────────────────────┘
      │
      ├── 本地基座：Qwen2.5-7B-Base（Ollama，可选）
      ├── 检索索引：memory / FAISS / Chroma
      └── 审计：audit/log.jsonl（哈希链）+ audit/snapshot/
```

**降级原则**：Ollama 不可用时，问答自动降级为“知识库原文抽取”，并显式标注
“未连接基座、未做生成式扩写”，绝不编造。

---

## 3. 目录结构

```text
.
├── config/
│   ├── config.yaml                  # 主配置（权限矩阵、沙盘权重、训练模板、审计）
│   ├── vector_store.faiss.yaml      # FAISS 片段示例
│   └── vector_store.chroma.yaml     # Chroma 片段示例
├── data/
│   ├── kb_seed.json                 # 32 篇种子文档 + 197 个关键词（7 个分类）
│   └── keywords_extra.txt           # 增量词库示例（导入用）
├── yuanyandao/                      # Python 包（零第三方依赖可运行）
│   ├── dispatcher.py                # 主循环与意图分流
│   ├── vector_store.py              # 双路检索、索引落盘、热重载
│   ├── keywords.py / embeddings.py  # 词典匹配 / 哈希向量器
│   ├── permissions.py               # 五级权限 + 动态密钥
│   ├── audit.py                     # JSONL 哈希链 + 快照
│   ├── sandbox.py                   # 并行分支、风险打分、冲突收敛
│   ├── train_scheduler.py           # QLoRA 配置 / 脚本 / 数据方案
│   ├── rag.py / documents.py        # RAG 生成与交付模板
│   ├── server.py / mcp_server.py    # HTTP / MCP 接入
│   └── cli.py                       # 命令行入口
├── scripts/                         # 入库、建索引、签发密钥、自检
├── tests/                           # 用例 JSON + 执行器
├── docs/test_report_v7.md           # 回归测试报告（可重复生成）
├── dsh-yuanyandao/                  # DSH 本地 bundle：MCP 接入 + 图标 + 中英文案
├── system_prompt.md                 # 基座 System Prompt
├── CHANGELOG.md                     # V7.0 变更记录
├── run_rtx5060.ps1 / .bat           # 一键启动
└── audit/ vector_db/ state/ train_output/   # 运行时生成
```

---

## 4. 快速开始

### 4.1 环境

- Python **3.10+**（开发验证：3.10.11 / Windows 11）
- 运行插件本体**不需要**任何第三方包；`numpy`、`faiss`、`chromadb` 均为可选加速
- Ollama 可选（不装则走抽取模式）

```powershell
cd C:\Users\zxd79\Documents\deepseek-harness\default-workspace
$env:PYTHONIOENCODING="utf-8"
Copy-Item .env.example .env      # 可选：写入 YYDS_SECRET / YYDS_ADMIN_TOKEN（启动时自动加载）

python -m yuanyandao health                     # 自检：知识库/Ollama/审计链
python -m yuanyandao "导出全部TA关键词" -l 4 --auto-key -f keywords
```

`--auto-key` 仅用于本机开发（用同一服务端密钥就地签发）；生产环境请用
`scripts/issue_key.py` 签发并显式传 `-k`。

### 4.2 一键脚本（RTX5060）

```powershell
.\run_rtx5060.ps1            # 自检 + 六条内置指令回归
.\run_rtx5060.ps1 -Serve     # 启动 HTTP 服务（默认 127.0.0.1:18080）
```

```bat
run_rtx5060.bat test
run_rtx5060.bat serve
```

### 4.3 依赖（可选）

```bash
pip install -r requirements.txt          # 默认仅注释项，不装任何包
pip install faiss-cpu numpy              # 启用 FAISS
pip install chromadb                     # 启用 Chroma
pip install pyyaml                       # 更快的 YAML 解析（缺失时用内置 miniyaml）
```

---

## 5. 六条内置调试指令

| # | 指令 | 预期 |
| --- | --- | --- |
| 1 | `导出全部TA关键词` | 逗号分隔关键词串（含 渲染管线/PBR/Nanite…） |
| 2 | `沙盘推演 RTX5060 4bit QLoRA训练基座风险` | 5 分支风险权重表 + 冲突 + 收敛结论 |
| 3 | `生成元衍智枢V7.0 BP Markdown文档` | BP 结构化文档 |
| 4 | `权限测试-无密钥访问TA知识库` | 标准化权限拒绝（`PERMISSION_DENIED`） |
| 5 | `刷新词库` | 热重载完成提示 + 索引指纹（L4） |
| 6 | `生成RTX5060基座微调yaml配置` | `qlora_config.yaml` + 训练脚本 + 数据清洗方案 |

```powershell
python -m yuanyandao "沙盘推演 RTX5060 4bit QLoRA训练基座风险" -l 4 --auto-key -f markdown
```

---

## 6. 五级权限与动态密钥

| 等级 | 名称 | 能力 |
| --- | --- | --- |
| 0 | 访客 | 仅公开知识库问答 |
| 1 | 基础用户 | TA / 模型 / 硬件知识库问答、文档交付 |
| 2 | 高级用户 | + 沙盘推演、项目资产库 |
| 3 | 开发操作员 | + 训练配置生成、快照导出 |
| 4 | 内核管理员 | + 词库入库、权限变更、全量快照 |

知识库按等级隔离：`public=0`、`ta/llm/hardware=1`、`project/sandbox_rules=2`、
`permission_rules=3`。**越权判定发生在检索之前**，拒绝时不返回任何受限内容。

动态密钥格式：

```text
YYDS.{level}.{expires_epoch}.{nonce}.{hmac_sha256[:24]}
```

```powershell
python scripts/issue_key.py --level 2 --ttl 3600
python -m yuanyandao "启动沙盘推演" -l 2 -k "YYDS.2....."
```

签名使用 `YYDS_SECRET`（环境变量优先）或 `permission.key_secret`；`key_single_use: true`
时启用防重放，同一 nonce 只能使用一次。

---

## 7. 词库维护与热重载

```powershell
# 批量入库（逗号或换行分隔，自动去重并重建索引）
python scripts/import_keywords.py --category ta --terms "路径追踪,体积云,次表面散射"
python scripts/import_keywords.py --category llm --file data/keywords_extra.txt

# 同时入库一段文档
python scripts/import_keywords.py --category hardware --terms "显存碎片" `
  --doc-title "显存碎片治理" --doc-content "开启 expandable_segments 可缓解碎片..." --doc-tags "显存,OOM"
```

运行期触发词：`刷新词库`（L4 热重载）、`入库 <词条>`（L4 追加）、`整理文档`（盘点）。

---

## 8. 训练调度（RTX5060 8G）

默认模板：Qwen2.5-7B-Base + 4bit NF4 双重量化 + LoRA r=8/alpha=16 + seq 1024 +
batch 2 × 累积 4 + `paged_adamw_8bit` + 梯度检查点 + bf16。

```powershell
python -m yuanyandao "生成RTX5060基座微调yaml配置" -l 4 --auto-key -f markdown
```

产物：

```text
train_output/qwen25-7b-qlora-<时间>/
├── qlora_config.yaml     # 可直接 yaml.safe_load
├── train.py              # SFT / DPO 双分支，--dry-run 先验证显存
└── dataset_plan.json     # 数据清洗与格式化方案
```

```powershell
# 先 dry-run，确认不 OOM 再正式训练
python train_output/qwen25-7b-qlora-*/train.py --config train_output/qwen25-7b-qlora-*/qlora_config.yaml --dry-run
```

显存估算（内置公式，输出在结果 `data.vram`）：

| 组成 | 估算 |
| --- | --- |
| 4bit 权重 | 参数量 × 0.5B × 1.06 ≈ 4.0 GB（7B） |
| LoRA + 优化器 | 0.05 GB 量级 |
| 激活 | 随 seq/batch 线性增长，梯度检查点约降 2.5× |
| logits | 词表 152k × seq × batch × 2B |

估算峰值 > 7.6 GB 时结果会提示降级：**seq↓ → batch↓+累积↑ → r↓ → CPU 卸载 → 换 3B**。

---

## 9. Ollama 对接

本机已安装 `qwen2.5:7b-instruct`（4.7GB）与 `qwen3-embedding:0.6b`，配置默认已对齐：

```powershell
ollama serve                       # 或直接启动 Ollama 桌面端
ollama list                        # 确认模型存在
ollama pull qwen2.5:7b-instruct    # 未安装时拉取
```

```yaml
hardware:
  ollama_endpoint: "http://127.0.0.1:11434"
  ollama_model: "qwen2.5:7b-instruct"
  embed_model: "qwen3-embedding:0.6b"
  model_mode: "instruct"           # instruct=对话式；base=续写式
  num_ctx: 4096                    # 8G 显存建议值
```

接口：`GET /api/tags` 健康检查、`POST /api/generate` 生成、`POST /api/embeddings` 向量。
不可用时 `runtime.offline_fallback: true` 自动降级为抽取模式。

实测（RTX5060 Laptop 8G，4bit Q4_K_M 全量上卡 29/29 层，占用约 4.5GB）：
首次调用需加载模型约 **31 秒**，之后模型驻留、普通问答 **5–10 秒**（218 token 生成约 5.7 秒）。
若觉得慢，可把 `max_new_tokens` 降到 512，或缩短 `system_prompt.md`。
需要真语义向量时，把 `vector_store.embed_provider` 改为 `ollama` 后发送【刷新词库】重建索引。

---

## 10. 服务化接入

### 10.1 HTTP

```powershell
python -m yuanyandao serve --host 127.0.0.1 --port 18080
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 存活检查 |
| GET | `/v1/status` | 知识库、Ollama、审计链、权限矩阵 |
| GET | `/v1/schema` | 工具 schema（供 DSH 注册） |
| GET | `/v1/keywords?category=ta` | 关键词导出 |
| POST | `/v1/invoke` | `{text,user_id,permission_level,api_key,format}` |
| POST | `/v1/reload` | 热重载（需 L4 密钥） |
| POST | `/v1/key` | 签发密钥（本机或 `X-Admin-Token`） |

```powershell
$key = python scripts/issue_key.py --level 4 --json | ConvertFrom-Json | Select-Object -ExpandProperty key
$body = @{ text="沙盘推演 RTX5060 4bit QLoRA训练基座风险"; permission_level=4; api_key=$key; format="json" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:18080/v1/invoke -Method Post -Body $body -ContentType "application/json"
```

### 10.2 MCP（已安装到 DeepSeek Harness）

本工作区附带本地 bundle `dsh-yuanyandao/`，已通过 `plugin_manager install_bundle`
安装进当前 DSH profile（`desktop`）并生效，无需手工改配置。装好后智能体可直接调用：

| DSH 工具 | 作用 |
| --- | --- |
| `mcp__yuanyandao__invoke` | 问答 / 沙盘推演 / 训练配置 / 交付文档 / 词库维护 |
| `mcp__yuanyandao__reload` | 热重载词库与向量索引（L4） |
| `mcp__yuanyandao__status` | 知识库规模、Ollama、审计链状态 |

bundle 的 `cordis.patch.yml` 实际内容（stdio 连接）：

```yaml
- insert:
    - id: yuanyandao-mcp
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: yuanyandao
        transport: stdio
        command: "C:/Users/zxd79/AppData/Local/Programs/Python/Python310/python.exe"
        args: ["-m", "yuanyandao", "mcp"]
        cwd: "C:/Users/zxd79/Documents/deepseek-harness/default-workspace"
        env: { YYDS_MCP_LEVEL: "4", YYDS_MCP_TRUSTED: "1", PYTHONIOENCODING: "utf-8" }
        toolCallTimeoutMs: 180000
        failOnStartupError: true
```

`YYDS_MCP_TRUSTED=1` 时，MCP 进程启动会用本地 `permission.key_secret`
就地签发一枚 L4 密钥（30 天，仅存于进程内），因此智能体无需传密钥即可调用。
想收紧权限，把 `YYDS_MCP_LEVEL` 改为 `2`/`3` 后重载插件即可。

手动验证连接：

```powershell
python -m yuanyandao mcp --level 4 --trusted     # 终端直接跑 stdio
```

卸载：在 DSH 插件管理中移除 `@local/yuanyandao-mcp`（`plugin_manager remove_bundle`），
不会删除工作区工程。

---

## 11. 审计日志与快照

- `audit/log.jsonl`：一行一次调用，字段见 `audit.log_fields`
  （user_id、permission_level、input、output、hit_keywords、source_doc、timestamp、branch_id、op_code），
  另含 `request_id`、`status`、`code`、`latency_ms`、`prev_hash`、`record_hash`。
- 哈希链：`record_hash = sha256(prev_hash + 规范化记录)`，任何一条被改都会让链校验失败。
- `audit/snapshot/`：沙盘与训练类调用保存完整请求/响应快照。

```powershell
python -m yuanyandao health        # 内含 audit.verify()
```

---

## 12. 测试

```powershell
python tests/run_tests.py                                   # 全量回归
python tests/run_tests.py --report docs/test_report_v7.md   # 同时生成测试报告
```

覆盖 12 条用例（含六条内置指令）+ 12 项内核自检：

```text
用例：12/12 通过　自检：12/12 通过
```

自检项：

| 自检 | 验证内容 |
| --- | --- |
| 审计哈希链 | 逐条重算 `record_hash` 与前向链接 |
| 词库热重载 | 临时目录真实新增词条/文档并重建索引 |
| 四格式输出 | Markdown / JSON / CSV / 逗号关键词 |
| 训练脚本 | 生成脚本可被 `ast.parse`，YAML 字段完整 |
| YAML 兜底 | 内置 miniyaml 解析权重、标量、行内列表 |
| 密钥生命周期 | 重放 / 过期 / 越级 / 签名篡改 / 缺失 5 类拦截 |
| 锚点锁定 | 改配置锚点后启动直接拒绝 |
| 审计防篡改 | 改动一条记录即被哈希链检出 |
| 后端回落 | faiss / chroma 缺失时回落 memory 且检索可用 |
| Ollama 生成 | 本地 Mock 服务验证上下文注入、生成模式与引用 |
| MCP 协议 | initialize / tools/list / tools/call / ping |
| HTTP 接口 | health/schema/key/invoke/reload/400 |

已生成的报告：[docs/test_report_v7.md](docs/test_report_v7.md)。

---

## 13. 关键配置

| 配置 | 说明 |
| --- | --- |
| `vector_store.engine` | `memory`（默认）/ `faiss` / `chroma` |
| `vector_store.dual_retrieval` | 关键词 + 语义双路开关 |
| `vector_store.min_score` | 未收录阈值，默认 0.18 |
| `permission.enable_dynamic_key` | 动态密钥开关 |
| `sandbox.max_branch` | 最大分支数，默认 5 |
| `sandbox.veto_threshold` | 单维度一票否决阈值，默认 80 |
| `train_scheduler.loss_type` | `dpo`（默认）/ `sft` |
| `audit.hash_chain` | 审计哈希链开关 |
| `safety.banned_patterns` | 输入/输出违禁片段 |

完整默认值见 `yuanyandao/config.py::DEFAULTS`。

---

## 14. 故障排查

| 现象 | 排查 |
| --- | --- |
| 启动报“锚点密钥校验失败” | `anchor_key` / `safety.anchor_lock` 必须为 `元衍道枢` |
| 所有问答都“知识库未收录” | 检查 `data/kb_seed.json`、`vector_store.min_score`、是否 `刷新词库` |
| 回答标注“抽取模式” | Ollama 未启动或模型未 pull；`python -m yuanyandao health` 查看原因 |
| L1 询问 TA 被拒 | 检查密钥等级与 `permission.kb_level` |
| 训练 OOM | 按 §8 降级顺序；先 `--dry-run` 看峰值 |
| 中文乱码（Windows） | `$env:PYTHONIOENCODING="utf-8"` |

---

## 15. 安全边界与已知限制

- 锚点锁定在代码与配置双层校验，配置被改会直接拒绝启动。
- 角色隔离采用“输出断言”而非模型自证：检测到“我是元衍智枢”等表述会标记违规。
- 哈希向量器是**离线可复现**方案，不是语义模型；需要更强语义召回时切
  `embed_provider: ollama` 并重建索引。
- Chroma 后端为兼容模式（维度一致性由本地向量保证），生产建议 FAISS 或默认 memory。
- 训练脚本依据 TRL 新版 API 编写，实际运行前请对齐本地 `trl` 版本。

---

_anchor: 元衍道枢 · version 7.0.0_
