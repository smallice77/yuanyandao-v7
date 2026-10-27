# 变更记录 · 元衍道枢 V7.0

本版本对应需求文档《元衍道枢 YuanYanDaoShu V7.0》，逐条落地如下。

## 1. 权限体系重构

- 五级权限标签（L0 访客 → L4 内核管理员）在 `config.yaml` 的 `permission.capabilities` 中数据化配置。
- 动态密钥 `YYDS.{level}.{exp}.{nonce}.{hmac}`，HMAC-SHA256 签名、有效期、可选一次性防重放。
- 知识库按等级隔离（`permission.kb_level`），**越权判定在检索之前**，拒绝时不返回受限内容。
- 实现：`yuanyandao/permissions.py`、`dispatcher.py::_capability_denied` / 越权命中判定。

## 2. 沙盘推演升级

- 多分支并行：`ThreadPoolExecutor` 并行构建，默认 5 支，覆盖保守 / 激进 / 替代 / 数据 / 外部五类取向。
- 反向溯因校验：每个分支对最高风险维度输出“现象 → 近因 → 根因”链。
- 冲突识别：按关键变量（批大小、序列长度、训练位置等）比对，登记冲突组与严重度。
- 冲突收敛：硬约束优先 + 单维度一票否决（默认 >80）+ 加权风险最低者胜出。
- 实现：`yuanyandao/sandbox.py`；风险维度与权重见 `config.yaml::sandbox.risk_dimensions`。

## 3. RAG 内核升级

- 双路检索：BM25 关键词召回 + 哈希向量语义召回，词典精确命中额外加权。
- 覆盖率阻尼：避免“流程/格式”这类泛词把无关文档顶到首位（V7 修复项）。
- 热更新：`刷新词库` / `scripts/build_index.py` 重建索引，无需重启进程。
- 来源标签与相关度阈值：低于 `min_score` 固定返回“知识库未收录”。
- 实现：`yuanyandao/vector_store.py`、`keywords.py`、`embeddings.py`。

## 4. 训练调度子单元

- 内置 RTX5060 8G 模板：4bit NF4 双重量化 + LoRA r=8 + seq 1024 + batch 2×累积 4。
- 产出 `qlora_config.yaml`、`train.py`（SFT/DPO 双分支，含 `--dry-run`）、`dataset_plan.json`。
- 显存估算公式化输出（权重 / 适配器 / 激活 / logits），超限时给出降级建议。
- 实现：`yuanyandao/train_scheduler.py`。

## 5. 输出规范

- 四格式统一渲染：Markdown / JSON / CSV / 逗号关键词，CLI `-f`、HTTP `format`、MCP `format` 均可切换。
- JSON 信封含 `status/intent/anchor/role/answer/data/citations/warnings/meta` 固定字段。
- 实现：`yuanyandao/formats.py`。

## 6. 审计升级

- JSONL 逐条落盘，字段覆盖需求 `audit.log_fields`，另加 `request_id/status/code/latency_ms`。
- 哈希链：`record_hash = sha256(prev_hash + 规范化记录)`，可用 `health` 一键校验。
- 沙盘 / 训练调用额外写完整快照到 `audit/snapshot/`。
- 实现：`yuanyandao/audit.py`。

## 7. 边界隔离

- 锚点“元衍道枢”在代码常量 + 配置双层校验，配置被改直接拒绝启动。
- 输入侧拦截锚点篡改与提示注入；输出侧检测角色混淆（“我是元衍智枢”等）。
- 基座 System Prompt 明确“调度内核 ≠ 业务中台”。
- 实现：`yuanyandao/anchors.py`、`system_prompt.md`。

## 8. 交付后加固（回归覆盖）

在初版交付基础上补齐了未覆盖分支，自检从 7 项扩展到 12 项：

| 加固项 | 方式 |
| --- | --- |
| Ollama 生成链路 | 测试内起本地 Mock `/api/tags` + `/api/generate`，断言上下文注入、`ollama:` 模式、引用不丢失 |
| 密钥生命周期 | 一次性密钥重放、过期、等级不一致、签名篡改、缺失 5 类均断言拦截 |
| 锚点锁定 | 复制配置改写 `anchor_key`，断言启动即抛错 |
| 审计防篡改 | 改动日志首条 `output`，断言哈希链校验失败 |
| 后端回落 | `faiss` / `chroma` 请求后端在依赖缺失时回落 memory，且检索仍可用 |
| HTTP 管理接口 | `/v1/schema`、`/v1/key`、`/v1/reload`、空 text 400 全覆盖 |
| 测试报告 | `run_tests.py --report` 生成 `docs/test_report_v7.md` |

当前结果：**用例 12/12，自检 12/12**。

## 8. DeepSeek Harness 安装

- 新增本地 bundle `dsh-yuanyandao/`（`@local/yuanyandao-mcp`），通过
  `@deepseek-ai/dsh-mcp-client` 以 stdio 方式接入当前 DSH profile。
- 工具名遵循 DSH 约定：`mcp__yuanyandao__invoke` / `__reload` / `__status`。
- 受信模式（`YYDS_MCP_TRUSTED=1`）在 MCP 进程启动时用本地密钥签发一枚
  30 天 L4 密钥，智能体无需在对话中传密钥；可通过 `YYDS_MCP_LEVEL` 收紧等级。
- 安装结果：`application: applied`，工具已注册并完成 invoke / reload / status
  三条链路的在线调用验证（2026-10-27）。

## V7.0 交付清单完成情况

| 要求 | 状态 | 位置 |
| --- | --- | --- |
| 插件主程序 | ✅ | `yuanyandao/` |
| 向量库 YAML（FAISS/Chroma） | ✅ | `config/` |
| 基座 System Prompt | ✅ | `system_prompt.md` |
| 关键词批量入库脚本 | ✅ | `scripts/import_keywords.py` |
| 沙盘并行分支 + 风险打分 | ✅ | `yuanyandao/sandbox.py` |
| 五级权限 + 动态密钥 | ✅ | `yuanyandao/permissions.py` |
| 审计 JSONL + 快照 | ✅ | `yuanyandao/audit.py` |
| README + 一键启动 + Ollama 示例 | ✅ | `README.md`、`run_rtx5060.*` |
| 测试用例 JSON | ✅ | `tests/` |
| DSH 插件安装 | ✅ | `dsh-yuanyandao/`（已安装并在线验证） |

## 已知限制

- 哈希向量器为离线可复现方案，语义能力弱于真嵌入；可切 `embed_provider: ollama`。
- Chroma 以兼容模式接入（向量仍由本地 embedder 计算）。
- 训练脚本面向 TRL 新版 API，需按本地版本微调参数名。
