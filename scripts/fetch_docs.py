#!/usr/bin/env python
"""把公开发布的网页文档抓成本项目可用的 markdown。

设计原则（重要）：
    只做「搬运 + 排版」，不改写、不摘要、不补写。
    任何改写都会让训练数据脱离原文，直接破坏 4.1.2（忠实性）的语义。

做三件事：
    1. 抓正文（去脚本/样式/导航噪声）
    2. 按「第X章」「一、」插二级标题（项目的 load_sections 靠 ## 切章节）
    3. HTML 表格转成 markdown 表格（项目的 table_points 会再转述成自然语言）

用法：
    python scripts/fetch_docs.py --url <URL> --name 财务报销管理制度 \
        --doc-title "某大学报销管理办法" --source "某大学财务处 https://..."

    # 只预览不落盘
    python scripts/fetch_docs.py --url <URL> --name tmp --dry-run
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "docs"

BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "blockquote", "pre", "dd", "dt",
}
SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "form"}


class _Extractor(HTMLParser):
    """把 HTML 摊平成「行 + 表格」；不保留任何标签样式。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.buf: list[str] = []
        self.skip = 0
        self.table_depth = 0
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    # --- 文本 ---
    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in SKIP_TAGS:
            self.skip += 1
            return
        if tag == "table":
            self.table_depth += 1
            self.rows = []
            return
        if self.table_depth:
            if tag == "tr":
                self.row = []
            elif tag in ("td", "th"):
                self.cell = []
            return
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
            return
        if tag == "table" and self.table_depth:
            self.table_depth -= 1
            if self.rows:
                self._emit_table(self.rows)
            return
        if self.table_depth:
            if tag in ("td", "th") and self.cell is not None and self.row is not None:
                self.row.append(self._clean("".join(self.cell)))
                self.cell = None
            elif tag == "tr" and self.row is not None:
                if any(c for c in self.row):
                    self.rows.append(self.row)
                self.row = None
            return
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        if self.table_depth:
            if self.cell is not None:
                self.cell.append(data)
            return
        self.buf.append(data)

    # --- 内部 ---
    @staticmethod
    def _clean(text: str) -> str:
        # 单元格里的换行会把 markdown 表格切成两行，必须压成空格
        text = text.replace("\u00a0", " ").replace("\u3000", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _flush(self) -> None:
        if not self.buf:
            return
        line = self._clean("".join(self.buf))
        self.buf.clear()
        if line:
            self.lines.append(line)

    def _emit_table(self, rows: list[list[str]]) -> None:
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        self.lines.append("| " + " | ".join(rows[0]) + " |")
        self.lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
        for r in rows[1:]:
            self.lines.append("| " + " | ".join(c.replace("|", "/") for c in r) + " |")
        self.rows = []


CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十百]+[章编])\s*(.*)$")
CN_SECTION_RE = re.compile(r"^([一二三四五六七八九十]+)、\s*(.+)$")
NOISE_RE = re.compile(
    r"^(打印|关闭|返回|上一篇|下一篇|扫一扫|分享到|字体|【?字号|索引号|发布机构|"
    r"成文日期|发布日期|来源|责任编辑|浏览次数|网站地图|版权所有)"
)


def to_markdown(text_lines: list[str]) -> list[str]:
    """插章节标题、去掉页眉页脚噪声。"""
    out: list[str] = []
    for raw in text_lines:
        line = raw.strip()
        if not line or NOISE_RE.match(line):
            continue
        m = CHAPTER_RE.match(line)
        if m:
            out.append("")
            out.append(f"## {m.group(1)} {m.group(2)}".rstrip())
            out.append("")
            continue
        m = CN_SECTION_RE.match(line)
        if m and len(line) <= 20:          # 只把短行当标题，避免误伤正文
            out.append("")
            out.append(f"## {line}")
            out.append("")
            continue
        out.append(line)
    # 压缩多余空行
    cleaned: list[str] = []
    for line in out:
        if line == "" and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(line)
    return cleaned


def fetch(url: str, timeout: int = 40) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
            "Accept-Language": "zh-CN,zh;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset()
    for enc in filter(None, [charset, "utf-8", "gb18030"]):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def html_to_lines(page: str) -> list[str]:
    parser = _Extractor()
    parser.feed(page)
    parser.close()
    parser._flush()
    return parser.lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True, help="输出文件名（不含 .md），也是文档短名")
    ap.add_argument("--doc-title", default="", help="文档正式标题，写进 # 一级标题")
    ap.add_argument("--source", default="", help="来源说明（机构 + URL），写进 SOURCES.md")
    ap.add_argument("--start", default="", help="正文起点标记（子串），之前的内容（导航/面包屑）全部丢弃")
    ap.add_argument("--end", default="", help="正文终点标记（子串），之后的内容（页脚/相关链接）全部丢弃")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    page = fetch(args.url)
    lines = html_to_lines(page)
    if args.start:
        hit = next((i for i, ln in enumerate(lines) if args.start in ln), None)
        if hit is None:
            print(f"[失败] 找不到起点标记：{args.start}")
            return 1
        lines = lines[hit:]
    if args.end:
        hit = next((i for i, ln in enumerate(lines) if args.end in ln), None)
        if hit is not None:
            lines = lines[:hit]
    body = to_markdown(lines)
    if len(body) < 20:
        print(f"[失败] 只抽到 {len(body)} 行，页面可能是动态渲染或需要登录")
        return 1

    title = args.doc_title or args.name
    text = f"# {title}\n\n" + "\n".join(body) + "\n"
    n_chap = sum(1 for ln in body if ln.startswith("## "))
    chars = sum(len(b) for b in lines)

    if args.dry_run:
        print(f"行数 {len(body)}｜章节 {n_chap}｜字符 {chars}")
        print("---- 预览前 30 行 ----")
        print("\n".join(body[:30]))
        return 0

    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / f"{args.name}.md"
    out.write_text(text, encoding="utf-8")
    print(f"写出 {out.relative_to(ROOT)}｜{len(body)} 行｜{n_chap} 个章节｜{chars} 字符")

    if args.source:
        man = DOCS / "SOURCES.md"
        line = f"- `{args.name}.md` — {args.source}"
        old = man.read_text(encoding="utf-8") if man.exists() else "# 语料来源\n\n| 文件 | 来源 |\n| --- | --- |\n"
        if line not in old:
            man.write_text(old.rstrip() + "\n" + line + "\n", encoding="utf-8")
            print(f"来源已记录到 {man.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
