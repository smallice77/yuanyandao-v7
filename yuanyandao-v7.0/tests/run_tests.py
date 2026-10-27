# -*- coding: utf-8 -*-
"""元衍道枢 V7.0 回归测试执行器。

覆盖：关键词检索/导出、权限拦截、沙盘推演、训练调度、词库热更新、审计链、
密钥生命周期、锚点/审计篡改、FAISS/Chroma 回落、Ollama 生成链路（本地 Mock）、
四种输出格式、MCP 协议、HTTP 接口、YAML 兜底解析、训练脚本语法。

    python tests/run_tests.py
    python tests/run_tests.py --verbose
    python tests/run_tests.py --report docs/test_report_v7.md
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yuanyandao import formats, miniyaml  # noqa: E402
from yuanyandao.dispatcher import Engine  # noqa: E402


# ----------------------------------------------------------------------
# 断言工具
# ----------------------------------------------------------------------
def _get(data: Any, dotted: str) -> Any:
    node = data
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def _haystack(payload: Dict[str, Any]) -> str:
    return str(payload.get("answer") or "") + "\n" + json.dumps(payload.get("data") or {}, ensure_ascii=False)


def evaluate(case: Dict[str, Any], payload: Dict[str, Any]) -> List[str]:
    expect = case.get("expect") or {}
    errors: List[str] = []
    data = payload.get("data") or {}
    haystack = _haystack(payload)

    if "status" in expect and payload.get("status") != expect["status"]:
        errors.append("status=%r，期望 %r" % (payload.get("status"), expect["status"]))
    if "intent" in expect and payload.get("intent") != expect["intent"]:
        errors.append("intent=%r，期望 %r" % (payload.get("intent"), expect["intent"]))
    if "code" in expect:
        actual_code = data.get("code") if isinstance(data, dict) else None
        if actual_code != expect["code"]:
            errors.append("code=%r，期望 %r" % (actual_code, expect["code"]))
    for word in expect.get("must_contain", []):
        if word not in haystack:
            errors.append("缺少内容: %r" % word)
    for word in expect.get("must_not_contain", []):
        if word in haystack:
            errors.append("不应出现: %r" % word)
    if "min_answer_chars" in expect and len(str(payload.get("answer") or "")) < int(expect["min_answer_chars"]):
        errors.append("回答长度 %d < %d" % (len(str(payload.get("answer") or "")), expect["min_answer_chars"]))
    if "citations_min" in expect and len(payload.get("citations") or []) < int(expect["citations_min"]):
        errors.append("引用数不足: %d" % len(payload.get("citations") or []))
    for path, minimum in (expect.get("data_min") or {}).items():
        value = _get(data, path)
        if not isinstance(value, (int, float)) or value < minimum:
            errors.append("data.%s=%r，应 >= %r" % (path, value, minimum))
    for path, expected in (expect.get("data_equals") or {}).items():
        value = _get(data, path)
        if value != expected:
            errors.append("data.%s=%r，期望 %r" % (path, value, expected))
    for path in expect.get("data_nonempty", []):
        value = _get(data, path)
        if not value:
            errors.append("data.%s 为空" % path)
    for path, minimum in (expect.get("data_min_len") or {}).items():
        value = _get(data, path)
        if not isinstance(value, (list, dict, str)) or len(value) < minimum:
            errors.append("data.%s 长度不足 %d" % (path, minimum))
    yaml_text = str(_get(data, "train_plan.yaml") or "")
    for snippet in expect.get("yaml_contains", []):
        if snippet not in yaml_text:
            errors.append("YAML 缺少: %r" % snippet)
    files = _get(data, "train_plan.files") or _get(data, "files") or {}
    for key in expect.get("files_exist", []):
        path = files.get(key)
        if not path or not Path(str(path)).exists():
            errors.append("文件不存在: %s=%r" % (key, path))
    return errors


def _issue(engine: Engine, level: int) -> str:
    if level < 1:
        return ""
    return str(engine.issue_key(level, subject="test-runner")["key"])


def _make_project(transform=None) -> Path:
    """复制一份配置与种子库到临时目录，得到隔离的 Engine 运行环境。"""
    temp = Path(tempfile.mkdtemp(prefix="yyds-check-"))
    (temp / "config").mkdir()
    (temp / "data").mkdir()
    text = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    if transform is not None:
        text = transform(text)
    (temp / "config" / "config.yaml").write_text(text, encoding="utf-8")
    shutil.copy(ROOT / "data" / "kb_seed.json", temp / "data" / "kb_seed.json")
    return temp


def _http_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 20.0) -> Any:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ----------------------------------------------------------------------
# 用例
# ----------------------------------------------------------------------
def run_cases(engine: Engine, verbose: bool = False) -> Tuple[int, int, List[Dict[str, Any]]]:
    spec = json.loads((ROOT / "tests" / "test_cases_v7.json").read_text(encoding="utf-8"))
    cases = spec.get("cases", [])
    passed = 0
    results: List[Dict[str, Any]] = []
    for case in cases:
        level = int((case.get("user") or {}).get("level", 0))
        key = "" if case.get("no_key") else _issue(engine, level)
        fmt = str(case.get("format") or "json")
        payload = engine.invoke(
            str(case.get("cmd")),
            user_id=str((case.get("user") or {}).get("id", "tester")),
            level=level,
            api_key=key,
            fmt=fmt,
        )
        errors = evaluate(case, payload)
        ok = not errors
        passed += 1 if ok else 0
        results.append({"case": case, "payload": payload, "errors": errors, "ok": ok})
        mark = "PASS" if ok else "FAIL"
        print("[%s] #%02d %-18s %s" % (mark, case.get("id"), case.get("category", ""), case.get("cmd")))
        if errors:
            for err in errors:
                print("       ↳ %s" % err)
            if verbose:
                print(json.dumps(payload, ensure_ascii=False, indent=2)[:3000])
    return passed, len(cases), results


# ----------------------------------------------------------------------
# 自检项
# ----------------------------------------------------------------------
def check_audit(engine: Engine) -> Tuple[bool, str]:
    answer = engine.audit.verify()
    ok = bool(answer.get("ok"))
    return ok, "哈希链：%s（%s 条）" % (answer.get("message"), answer.get("checked"))


def check_hot_reload(engine: Engine) -> Tuple[bool, str]:
    temp = _make_project()
    try:
        local = Engine(config_path=str(temp / "config" / "config.yaml"))
        term = "元衍热重载测试词"
        if local.kb.lexicon.extract(term):
            return False, "测试词已存在，环境不干净"
        local.kb.add_keywords("ta", [term])
        doc = local.kb.add_document(
            "热重载验证文档", "本文档用于验证词库热重载：%s。" % term, category="ta", tags=[term]
        )
        local.kb.reload()
        if not local.kb.lexicon.extract(term):
            return False, "重载后关键词未进入词典"
        hit = local.kb.search(term, top_n=3, max_level=4)
        if doc.id not in [h.doc.id for h in hit.hits]:
            return False, "重载后新增文档未召回"
        return True, "新增 %s 并在重载后召回" % doc.id
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def check_formats(engine: Engine) -> Tuple[bool, str]:
    key = _issue(engine, 4)
    payload = engine.invoke("导出全部TA关键词", user_id="fmt", level=4, api_key=key, fmt="json")
    outputs = {fmt: formats.render(payload, fmt) for fmt in ("markdown", "json", "csv", "keywords")}
    problems = [name for name, text in outputs.items() if not text.strip()]
    if problems:
        return False, "空输出: %s" % ",".join(problems)
    if "," not in outputs["keywords"]:
        return False, "关键词格式不是逗号分隔"
    sandbox_md = formats.render(
        engine.invoke("沙盘推演 RTX5060 4bit QLoRA训练基座风险", user_id="fmt", level=4,
                      api_key=key, fmt="markdown"),
        "markdown",
    )
    if "| 分支" not in sandbox_md:
        return False, "沙盘 Markdown 缺少风险表"
    return True, "markdown/json/csv/keywords 均可用"


def check_train_script(engine: Engine) -> Tuple[bool, str]:
    plan = engine.train.build_plan()
    try:
        ast.parse(plan["script"])
    except SyntaxError as exc:
        return False, "训练脚本语法错误: %s" % exc
    if "load_in_4bit" not in plan["yaml"]:
        return False, "训练 YAML 缺少 4bit 配置"
    return True, "脚本可解析，YAML 字段完整"


def check_miniyaml() -> Tuple[bool, str]:
    text = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    data = miniyaml.loads(text)
    dims = data.get("sandbox", {}).get("risk_dimensions", [])
    if len(dims) != 6 or not isinstance(dims[0], dict):
        return False, "miniyaml 解析 risk_dimensions 失败"
    if abs(sum(float(d["weight"]) for d in dims) - 1.0) > 0.02:
        return False, "miniyaml 权重解析错误"
    if data.get("hardware", {}).get("base_model") != "qwen2.5-7b-base":
        return False, "miniyaml 解析标量失败"
    if data.get("permission", {}).get("level_list") != [0, 1, 2, 3, 4]:
        return False, "miniyaml 解析行内列表失败"
    return True, "零依赖 YAML 兜底解析通过"


def check_key_lifecycle() -> Tuple[bool, str]:
    temp = _make_project(lambda text: text.replace("key_single_use: false", "key_single_use: true"))
    try:
        local = Engine(config_path=str(temp / "config" / "config.yaml"))
        issued = local.issue_key(2)
        key = str(issued["key"])
        first = local.policy.authenticate("u", 2, key)
        second = local.policy.authenticate("u", 2, key)
        if not first.ok:
            return False, "首次校验失败: %s" % first.reason
        if second.ok or second.code != "KEY_REPLAY":
            return False, "一次性密钥未拦截重放（%s）" % second.code
        expired = local.policy.authenticate("u", 2, str(local.issue_key(2, ttl=-5)["key"]))
        if expired.code != "KEY_EXPIRED":
            return False, "过期密钥未拦截（%s）" % expired.code
        mismatch = local.policy.verify_key(str(local.issue_key(3)["key"]), 2)
        if mismatch.get("code") != "LEVEL_MISMATCH":
            return False, "等级不一致未拦截（%s）" % mismatch.get("code")
        bad = key[:-1] + ("0" if key[-1] != "0" else "1")
        if local.policy.verify_key(bad, 2).get("code") != "BAD_SIGNATURE":
            return False, "签名被篡改未拦截"
        if local.policy.verify_key("", 2).get("code") != "MISSING_KEY":
            return False, "空密钥未拦截"
        return True, "重放/过期/越级/篡改/缺失 5 类均拦截"
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def check_anchor_guard() -> Tuple[bool, str]:
    temp = _make_project(lambda text: text.replace('anchor_key: "元衍道枢"', 'anchor_key: "被篡改"'))
    try:
        try:
            Engine(config_path=str(temp / "config" / "config.yaml"))
        except (ValueError, RuntimeError) as exc:
            if "锚点" in str(exc):
                return True, "配置锚点被改，启动即拒绝"
            return False, "异常类型正确但信息未提锚点: %s" % exc
        return False, "锚点被篡改仍成功启动"
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def check_audit_tamper() -> Tuple[bool, str]:
    temp = _make_project()
    try:
        local = Engine(config_path=str(temp / "config" / "config.yaml"))
        key = _issue(local, 1)
        local.invoke("导出全部TA关键词", user_id="tamper", level=1, api_key=key, fmt="json")
        log_path = temp / "audit" / "log.jsonl"
        lines = log_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return False, "审计日志为空"
        record = json.loads(lines[0])
        record["output"] = "被篡改的输出"
        lines[0] = json.dumps(record, ensure_ascii=False, sort_keys=True)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = local.audit.verify()
        if result.get("ok"):
            return False, "篡改审计记录后校验仍通过"
        return True, "篡改第 1 条后被哈希链检出"
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def check_backend_fallback() -> Tuple[bool, str]:
    has_faiss = importlib.util.find_spec("faiss") is not None
    has_chroma = importlib.util.find_spec("chromadb") is not None
    details = []
    for engine_name, expected in (
        ("faiss", "faiss" if has_faiss else "memory"),
        ("chroma", "chroma-compatible" if has_chroma else "memory"),
    ):
        temp = _make_project(lambda text, e=engine_name: text.replace('engine: "memory"', 'engine: "%s"' % e, 1))
        try:
            local = Engine(config_path=str(temp / "config" / "config.yaml"))
            actual = local.kb._backend
            if actual != expected:
                return False, "%s 后端期望 %s，实际 %s" % (engine_name, expected, actual)
            hit = local.kb.search("QLoRA 显存", top_n=3, max_level=4)
            if not hit.hits:
                return False, "%s 后端检索无结果" % engine_name
            details.append("%s→%s" % (engine_name, actual))
        finally:
            shutil.rmtree(temp, ignore_errors=True)
    return True, "后端回落正常：" + "，".join(details)


def check_ollama_generation() -> Tuple[bool, str]:
    captured: Dict[str, Any] = {}

    class MockOllama(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return

        def _send(self, payload: Dict[str, Any], code: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/tags":
                self._send({"models": [{"name": "qwen2.5:7b"}]})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {}
            captured["payload"] = payload
            captured["path"] = self.path
            self._send({
                "response": "根据资料[1]，Nanite 负责虚拟化几何与自动 LOD，Lumen 负责动态全局光照与反射。[1]",
                "eval_count": 42,
                "done": True,
            })

    mock = ThreadingHTTPServer(("127.0.0.1", 0), MockOllama)
    port = mock.server_address[1]
    thread = threading.Thread(target=mock.serve_forever, daemon=True)
    thread.start()
    temp = None
    try:
        def transform(text: str) -> str:
            return re.sub(
                r'ollama_endpoint:\s*".*?"',
                'ollama_endpoint: "http://127.0.0.1:%d"' % port,
                text,
                count=1,
            )

        temp = _make_project(transform)
        local = Engine(config_path=str(temp / "config" / "config.yaml"))
        key = _issue(local, 1)
        payload = local.invoke(
            "UE5 的 Nanite 和 Lumen 分别解决什么问题",
            user_id="rag", level=1, api_key=key, fmt="json",
        )
        data = payload.get("data") or {}
        if payload.get("status") != "ok":
            return False, "生成链路状态异常: %s" % payload.get("status")
        if data.get("degraded"):
            return False, "Mock 基座可用时仍降级"
        if not str(data.get("mode", "")).startswith("ollama:"):
            return False, "未走 Ollama 生成: %s" % data.get("mode")
        if not payload.get("citations"):
            return False, "生成结果缺少引用"
        prompt = str((captured.get("payload") or {}).get("prompt") or "")
        if "参考资料" not in prompt or "Nanite" not in prompt:
            return False, "提示词未注入检索上下文"
        if "Nanite" not in str(payload.get("answer") or ""):
            return False, "回答未包含模型输出"
        return True, "Mock Ollama 生成 + 上下文注入 + 引用正常"
    finally:
        mock.shutdown()
        mock.server_close()
        thread.join(timeout=3)
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)


def check_mcp(engine: Engine) -> Tuple[bool, str]:
    from yuanyandao.mcp_server import MCPServer

    server = MCPServer(engine)
    init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    if not init or init["result"]["serverInfo"]["name"] != "yuanyandao":
        return False, "initialize 返回异常"
    tools = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = [tool["name"] for tool in tools["result"]["tools"]]
    if names != ["invoke", "reload", "status"]:
        return False, "工具名不符合 DSH 短名约定: %s" % names
    key = _issue(engine, 4)
    call = server.handle({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "invoke", "arguments": {
            "text": "导出全部TA关键词", "permission_level": 4, "api_key": key, "format": "json"}},
    })
    if call["result"].get("isError") or call["result"]["structuredContent"]["status"] != "ok":
        return False, "tools/call 返回错误"
    trusted = MCPServer(engine, default_level=4, trusted=True)
    trusted_call = trusted.handle({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "invoke", "arguments": {"text": "导出全部TA关键词", "format": "json"}},
    })
    if trusted_call["result"].get("isError") or trusted_call["result"]["structuredContent"]["status"] != "ok":
        return False, "受信模式默认等级调用失败"
    ping = server.handle({"jsonrpc": "2.0", "id": 5, "method": "ping", "params": {}})
    if "result" not in ping:
        return False, "ping 返回异常"
    unknown = server.handle({"jsonrpc": "2.0", "id": 6, "method": "no_such_method", "params": {}})
    if unknown.get("error", {}).get("code") != -32601:
        return False, "未知方法应返回 -32601"
    return True, "短名工具 / 受信模式 / tools/call / ping 正常"


def check_http(engine: Engine) -> Tuple[bool, str]:
    from yuanyandao.server import make_server

    httpd = make_server(engine, "127.0.0.1", 0)
    port = httpd.server_address[1]
    base = "http://127.0.0.1:%d" % port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        health = _http_json(base + "/health", timeout=5)
        if not health.get("ok"):
            return False, "/health 返回异常"
        schema = _http_json(base + "/v1/schema", timeout=5)
        if schema.get("schema", {}).get("name") != "yuanyandao_invoke":
            return False, "/v1/schema 返回异常"
        issued = _http_json(base + "/v1/key", {"level": 2}, timeout=5)
        if not issued.get("ok") or int(issued["key"]["level"]) != 2:
            return False, "/v1/key 签发异常"
        key = _issue(engine, 4)
        body = {
            "text": "生成RTX5060基座微调yaml配置",
            "permission_level": 4,
            "api_key": key,
            "format": "json",
        }
        result = _http_json(base + "/v1/invoke", body)
        if not result.get("ok") or result["result"]["intent"] != "train":
            return False, "/v1/invoke 返回异常"
        reload_resp = _http_json(base + "/v1/reload", {"api_key": key, "permission_level": 4})
        if not reload_resp.get("ok"):
            return False, "/v1/reload 返回异常"
        try:
            _http_json(base + "/v1/invoke", {"text": ""})
            return False, "空 text 未返回 400"
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                return False, "空 text 返回 %d，期望 400" % exc.code
        return True, "health/schema/key/invoke/reload/400 全部正常（端口 %d）" % port
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def write_report(path: Path, passed: int, total: int, results: List[Dict[str, Any]],
                 checks: List[Tuple[str, Tuple[bool, str]]]) -> None:
    check_passed = sum(1 for _, (ok, _) in checks if ok)
    lines = [
        "# 元衍道枢 V7.0 测试报告",
        "",
        "- 生成时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "- 运行环境：Python %s ｜ %s" % (sys.version.split()[0], platform.platform()),
        "- 执行命令：`python tests/run_tests.py`",
        "- 结论：**用例 %d/%d 通过，自检 %d/%d 通过**" % (passed, total, check_passed, len(checks)),
        "",
        "## 一、功能用例",
        "",
        "| # | 类别 | 指令 | 结果 |",
        "| --- | --- | --- | --- |",
    ]
    for item in results:
        case = item["case"]
        lines.append(
            "| %s | %s | `%s` | %s |"
            % (case.get("id"), case.get("category"), case.get("cmd"), "PASS" if item["ok"] else "FAIL")
        )
    lines += [
        "",
        "## 二、内核自检",
        "",
        "| 自检项 | 结果 | 说明 |",
        "| --- | --- | --- |",
    ]
    for name, (ok, detail) in checks:
        lines.append("| %s | %s | %s |" % (name, "PASS" if ok else "FAIL", detail))
    lines += [
        "",
        "## 三、覆盖说明",
        "",
        "- 权限：L0–L4 能力矩阵、知识库等级隔离、动态密钥重放/过期/越级/篡改/缺失。",
        "- 检索：关键词 + 向量双路、未收录兜底、覆盖率阻尼、热重载与增量入库。",
        "- 沙盘：5 分支并行、六维加权、一票否决、冲突登记与硬约束收敛。",
        "- 训练：QLoRA YAML/脚本/数据方案生成，脚本语法与 8G 显存估算。",
        "- 审计：JSONL 字段完整性、哈希链校验、篡改检出、快照归档。",
        "- 接入：CLI、HTTP（含管理接口）、MCP stdio、四格式渲染。",
        "- 生成：Mock Ollama 验证上下文注入与生成模式（真实基座需本机启动 Ollama）。",
        "",
        "## 四、已知限制",
        "",
        "- 未安装 Ollama 时问答走知识库抽取模式，属预期降级。",
        "- FAISS/Chroma 未安装时自动回落 memory 后端，报告中会标注实际后端。",
        "- 训练脚本按 TRL 新版 API 生成，执行前需对齐本地 `trl` 版本。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="元衍道枢 V7.0 回归测试")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report", default="", help="把测试报告写到指定 Markdown 路径")
    args = parser.parse_args()

    print("=" * 78)
    print("元衍道枢 YuanYanDaoShu V7.0 · 回归测试（锚点：元衍道枢）")
    print("=" * 78)
    engine = Engine()
    passed, total, results = run_cases(engine, verbose=args.verbose)

    checks = [
        ("审计哈希链", check_audit(engine)),
        ("词库热重载", check_hot_reload(engine)),
        ("四格式输出", check_formats(engine)),
        ("训练脚本", check_train_script(engine)),
        ("YAML 兜底", check_miniyaml()),
        ("密钥生命周期", check_key_lifecycle()),
        ("锚点锁定", check_anchor_guard()),
        ("审计防篡改", check_audit_tamper()),
        ("后端回落", check_backend_fallback()),
        ("Ollama 生成", check_ollama_generation()),
        ("MCP 协议", check_mcp(engine)),
        ("HTTP 接口", check_http(engine)),
    ]
    print("-" * 78)
    check_passed = 0
    for name, (ok, detail) in checks:
        check_passed += 1 if ok else 0
        print("[%s] %-10s %s" % ("PASS" if ok else "FAIL", name, detail))

    print("-" * 78)
    print("用例：%d/%d 通过　自检：%d/%d 通过" % (passed, total, check_passed, len(checks)))
    failed = [r for r in results if not r["ok"]]
    if failed:
        print("失败用例：" + "、".join("#%s" % r["case"].get("id") for r in failed))
    print("=" * 78)

    if args.report:
        write_report(Path(args.report), passed, total, results, checks)
        print("测试报告已写入 %s" % args.report)
    return 0 if passed == total and check_passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
