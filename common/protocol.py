"""统一输出协议 v1：构造与解析。

协议形态（详见 config/prompts/protocol.txt）：
  {"type":"tool_call","name":..., "arguments":{...}}
  {"type":"clarify","missing":[...], "reply":"..."}
  {"type":"refuse","reason":"permission","reply":"..."}
  {"type":"answer","reply":"... [1][2]"}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def make_tool_call(name: str, arguments: dict) -> str:
    return dumps({"type": "tool_call", "name": name, "arguments": arguments or {}})


def make_clarify(missing: list[str], reply: str) -> str:
    return dumps({"type": "clarify", "missing": list(missing), "reply": reply})


def make_refuse(reply: str, reason: str = "permission") -> str:
    return dumps({"type": "refuse", "reason": reason, "reply": reply})


def make_answer(reply: str) -> str:
    return dumps({"type": "answer", "reply": reply})


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> dict | None:
    """从可能夹带自然语言的输出中抽取第一个完整 JSON 对象。"""
    if not text:
        return None
    cleaned = _strip_fences(text)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        start = -1
                        continue
                    if isinstance(obj, dict):
                        return obj
                    start = -1
    return None


@dataclass
class Parsed:
    kind: str = "invalid"          # tool_call | clarify | refuse | answer | invalid
    name: str | None = None
    arguments: dict = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    reply: str = ""
    citations: list[int] = field(default_factory=list)
    json_valid: bool = False
    raw: str = ""
    source: str = "text"           # text | native_tool_calls

    @property
    def has_tool_call(self) -> bool:
        return self.kind == "tool_call" and bool(self.name)


def _normalize_native(call: Any) -> tuple[str | None, dict]:
    if isinstance(call, dict):
        fn = call.get("function") or call
        name = fn.get("name")
        args = fn.get("arguments")
    else:  # 对象形态
        fn = getattr(call, "function", None) or call
        name = getattr(fn, "name", None)
        args = getattr(fn, "arguments", None)
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {"__raw__": args}
    elif not isinstance(args, dict):
        args = {}
    return name, args


def parse_output(text: str, native_tool_calls: list | None = None) -> Parsed:
    raw = text or ""
    if native_tool_calls:
        name, args = _normalize_native(native_tool_calls[0])
        if name:
            return Parsed(
                kind="tool_call",
                name=name,
                arguments=args,
                json_valid=True,
                raw=raw,
                source="native_tool_calls",
            )

    obj = extract_json_object(raw)
    if obj is None:
        # 纯文本回复：按 answer 处理（不回填为 invalid，便于 B 层统计）
        plain = _strip_fences(raw)
        return Parsed(
            kind="answer" if plain else "invalid",
            reply=plain,
            citations=[int(n) for n in CITATION_RE.findall(plain)],
            json_valid=False,
            raw=raw,
        )

    kind = (obj.get("type") or "").strip().lower()
    reply = str(obj.get("reply") or obj.get("content") or "").strip()
    parsed = Parsed(kind=kind or "invalid", reply=reply, json_valid=True, raw=raw,
                    citations=[int(n) for n in CITATION_RE.findall(reply)])
    if kind == "tool_call":
        parsed.name = obj.get("name")
        args = obj.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        parsed.arguments = args if isinstance(args, dict) else {}
    elif kind == "clarify":
        miss = obj.get("missing") or []
        parsed.missing = [m for m in miss if isinstance(m, str)] if isinstance(miss, list) else []
    if kind not in {"tool_call", "clarify", "refuse", "answer"}:
        parsed.kind = "invalid"
    return parsed
