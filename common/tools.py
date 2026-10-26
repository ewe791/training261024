"""工具 schema 与 System Prompt 装配。"""
from __future__ import annotations

import json
from functools import lru_cache

from common.config import CONFIG_DIR, PROMPT_DIR

AGENT_TOOLS = {
    "risk": "tools_risk.json",
    "kb_admin": "tools_kb_admin.json",
    "kb_user": None,
}
AGENT_PROMPTS = {
    "risk": "risk_system.txt",
    "kb_admin": "kb_admin_system.txt",
    "kb_user": "kb_user_system.txt",
}


@lru_cache(maxsize=None)
def load_tools(agent: str) -> list[dict]:
    fname = AGENT_TOOLS.get(agent)
    if not fname:
        return []
    return json.loads((CONFIG_DIR / fname).read_text(encoding="utf-8"))


def tool_names(agent: str) -> list[str]:
    return [t["function"]["name"] for t in load_tools(agent)]


@lru_cache(maxsize=None)
def required_params(agent: str, tool: str) -> list[str]:
    for t in load_tools(agent):
        fn = t["function"]
        if fn["name"] == tool:
            return list(fn.get("parameters", {}).get("required", []))
    return []


@lru_cache(maxsize=None)
def all_params(agent: str, tool: str) -> list[str]:
    for t in load_tools(agent):
        fn = t["function"]
        if fn["name"] == tool:
            return list(fn.get("parameters", {}).get("properties", {}).keys())
    return []


@lru_cache(maxsize=None)
def protocol_text() -> str:
    return (PROMPT_DIR / "protocol.txt").read_text(encoding="utf-8").strip()


@lru_cache(maxsize=None)
def base_prompt(agent: str) -> str:
    name = AGENT_PROMPTS[agent]
    head = (PROMPT_DIR / name).read_text(encoding="utf-8").strip()
    return head + "\n\n" + protocol_text()


def fragments_block(contexts: list[dict]) -> str:
    lines = ["【知识片段】"]
    for i, c in enumerate(contexts, start=1):
        title = c.get("title", "")
        lines.append(f"[{i}] {title}（{c.get('unit_id', '')}）")
        lines.append(c.get("text", "").strip())
        lines.append("")
    return "\n".join(lines).strip()


def format_hint(fmt: str | None) -> str:
    """把 4.1.1 的答案格式要求写进 System Prompt（训练与评测使用同一份提示）。"""
    if not fmt:
        return ""
    from common.formatting import load_formats

    spec = load_formats().get(fmt) or {}
    req = spec.get("required") or []
    if spec.get("ordered"):
        rule = f"至少 {spec.get('min_steps', 2)} 个有序步骤，每步以动词开头"
    else:
        rule = "必须按顺序包含：" + "、".join(req)
    return f"\n\n【输出格式要求】本次回答请严格使用《{fmt}》模板。{spec.get('desc', '')}\n{rule}"


def system_prompt(agent: str, contexts: list[dict] | None = None, fmt: str | None = None) -> str:
    text = base_prompt(agent)
    if contexts:
        text = text + "\n\n" + fragments_block(contexts)
    text += format_hint(fmt)
    return text
