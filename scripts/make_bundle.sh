#!/usr/bin/env bash
# 打包上云用的项目副本（只含代码/配置/源文档，不含生成物）。
#
# 用法：
#   bash scripts/make_bundle.sh              # 生成 dist/ft-kb-4.1-<日期>.tar.gz
#   bash scripts/make_bundle.sh --with-data  # 连生成好的 jsonl 一起打（约 +10 MB，云端省 10 秒）
#
# 设计取舍：默认不打 data/*.jsonl。数据 10 秒就能按固定 seed 重建，
# 而生成物进包会让「云端数字和本地不一致时无法判断是谁的问题」。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_DATA=0
[ "${1:-}" = "--with-data" ] && WITH_DATA=1

STAMP="$(date +%Y%m%d-%H%M)"
NAME="ft-kb-4.1-$STAMP"
mkdir -p dist

EXCLUDES=(
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.workbuddy'
  --exclude='.venv' --exclude='venv' --exclude='dist'
  --exclude='.git' --exclude='logs' --exclude='outputs'
  --exclude='eval/results_*' --exclude='eval/compare.json'
  --exclude='eval/split_meta.json' --exclude='reports/*'
  --exclude='eval/annotations/*' --exclude='.env'
)

if [ "$WITH_DATA" -eq 0 ]; then
  EXCLUDES+=(--exclude='data/*.jsonl' --exclude='data/raw/*.jsonl')
fi

tar czf "dist/$NAME.tar.gz" "${EXCLUDES[@]}" -C "$ROOT" \
  START_HERE.md README.md requirements.txt .env.example .gitignore \
  common config data/docs docs eval scripts tests train

echo "== 打包完成 =="
echo "包:    dist/$NAME.tar.gz"
echo "大小: $(du -h "dist/$NAME.tar.gz" | cut -f1)"
if command -v sha256sum >/dev/null 2>&1; then
  echo "校验: $(sha256sum "dist/$NAME.tar.gz" | cut -d' ' -f1)"
fi
echo
echo "== 内容清单（前 40 项）=="
tar tzf "dist/$NAME.tar.gz" | head -40
echo
echo "== 上传到云实例 =="
echo "  scp -P <端口> dist/$NAME.tar.gz root@<主机>:/root/autodl-tmp/"
echo "  # 或在实例上："
echo "  mkdir -p /root/autodl-tmp/ft-kb-4.1 && tar xzf /root/autodl-tmp/$NAME.tar.gz -C /root/autodl-tmp/ft-kb-4.1"
