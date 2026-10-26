#!/usr/bin/env bash
# 无卡模式（实例只有 CPU）把所有「不需要 GPU」的活一次做完：
#   装依赖 → 下基座模型 → 写 .env → 造数据 → 训练数据体检 → 单测 → 环境自检
#
# 用法（在项目根目录执行）：
#   bash scripts/nocard_prepare.sh
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct bash scripts/nocard_prepare.sh   # 先用小模型验证链路
#   SKIP_MODEL=1 bash scripts/nocard_prepare.sh                       # 模型已下好，只配环境
#   MODEL_DIR=/root/autodl-tmp/Qwen2.5-7B-Instruct bash scripts/nocard_prepare.sh
#
# 设计前提：项目目录、venv、模型都在**数据盘**（/root/autodl-tmp）。
# 换卡型/换主机时系统盘可能被重置，数据盘不会——这三个放数据盘就不用重来一遍。
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/${MODEL##*/}}"
VENV_DIR="${VENV_DIR:-$PROJ/.venv}"
SKIP_MODEL="${SKIP_MODEL:-0}"
SKIP_DATA="${SKIP_DATA:-0}"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
warn() { printf '\033[33m[注意] %s\033[0m\n' "$*"; }
die() { printf '\033[31m[失败] %s\033[0m\n' "$*"; exit 1; }

free_gb() {
  local g
  g=$(df -Pk "$1" 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}')
  echo "${g:-0}"
}
dir_gb() {
  local g
  g=$(du -sk "$1" 2>/dev/null | awk '{printf "%.0f", $1/1048576}')
  echo "${g:-0}"
}

cat <<EOF
============================================================
无卡模式环境准备
  项目目录 : $PROJ
  基座模型 : $MODEL
  模型目录 : $MODEL_DIR
  虚拟环境 : $VENV_DIR
============================================================
EOF

say "1/8 磁盘与路径检查"
mkdir -p "$MODEL_DIR"
case "$MODEL" in
  *7B*) NEED=20 ;;
  *4B*|*3B*) NEED=10 ;;
  *) NEED=6 ;;
esac
FREE_M="$(free_gb "$MODEL_DIR")"
FREE_P="$(free_gb "$PROJ")"
echo "模型目录所在分区剩余 ${FREE_M} GB｜项目目录所在分区剩余 ${FREE_P} GB｜按 $MODEL 估算需要 ${NEED} GB"
[ "${FREE_M:-0}" -ge "$NEED" ] || die "模型目录空间不足（需 ≥${NEED} GB）。换到数据盘：MODEL_DIR=/root/autodl-tmp/xxx bash scripts/nocard_prepare.sh"

case "$MODEL_DIR" in
  /root/autodl-tmp/*|/root/autodl-tmp) : ;;
  *) warn "模型目录不在 /root/autodl-tmp（数据盘）。放系统盘的话，换卡型后可能被重置，需要重新下 15 GB。" ;;
esac

say "2/8 安装依赖（不装 torch，不需要 GPU）"
VENV_DIR="$VENV_DIR" bash "$PROJ/scripts/cloud_setup.sh" --no-card --no-check
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || die "没找到 $PY，venv 创建失败"

say "3/8 确认 torch（无卡模式下 cuda.is_available=False 属正常）"
"$PY" - <<'PY'
import torch
print(f"torch {torch.__version__} | CUDA 编译版本 {torch.version.cuda or '无'} "
      f"| cuda.is_available={torch.cuda.is_available()}")
PY

say "4/8 下载基座模型"
if [ "$SKIP_MODEL" = "1" ]; then
  echo "SKIP_MODEL=1，跳过下载"
elif [ -f "$MODEL_DIR/config.json" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
  echo "已存在完整模型，跳过下载（大小 $(dir_gb "$MODEL_DIR") GB）"
else
  echo "开始下载（7B 约 15.2 GB，20-40 分钟；中断后重跑本脚本会续传）"
  if [ -x "$VENV_DIR/bin/modelscope" ]; then
    "$VENV_DIR/bin/modelscope" download --model "$MODEL" --local_dir "$MODEL_DIR"
  else
    "$PY" - "$MODEL" "$MODEL_DIR" <<'PY'
import sys
from modelscope import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
  fi
fi

echo "校验模型文件："
ls "$MODEL_DIR"/config.json >/dev/null 2>&1 || die "缺 config.json，下载不完整，重跑本脚本续传"
ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || die "缺 *.safetensors，下载不完整，重跑本脚本续传"
echo "  分片数 $(ls "$MODEL_DIR"/*.safetensors | wc -l) 个｜总大小 $(dir_gb "$MODEL_DIR") GB"
MSIZE="$(dir_gb "$MODEL_DIR")"
[ "$MSIZE" -ge 3 ] || warn "模型体积只有 ${MSIZE} GB，偏小，可能是中断的下载"

say "5/8 生成 .env"
"$PY" - "$PROJ" "$MODEL_DIR" <<'PY'
import pathlib
import sys

proj = pathlib.Path(sys.argv[1])
model_dir = sys.argv[2]
env = proj / ".env"
tpl = proj / ".env.example"

want = {
    "FTKB_BACKEND": "local",
    "FTKB_BASE_MODEL_PATH": model_dir,
    "FTKB_ADAPTER_PATH": str(proj / "outputs" / "lora-knowledge-agent-4.1"),
}

if env.exists():
    lines = env.read_text(encoding="utf-8").splitlines()
    out, seen = [], set()
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in want:
            seen.add(key)
            new = f"{key}={want[key]}"
            if line.strip() != new:
                print(f"  更新 {line.strip()}  ->  {new}")
            out.append(new)
        else:
            out.append(line)
    for key, val in want.items():
        if key not in seen:
            out.append(f"{key}={val}")
            print(f"  追加 {key}={val}")
    env.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"已更新 {env}（其余自定义项保留）")
else:
    text = tpl.read_text(encoding="utf-8")
    out, seen = [], set()
    for line in text.splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in want:
            seen.add(key)
            out.append(f"{key}={want[key]}")
        else:
            out.append(line)
    for key, val in want.items():
        if key not in seen:
            out.append(f"{key}={val}")
    env.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"已从 .env.example 生成 {env}")
print("  关键项：")
for k, v in want.items():
    print(f"    {k}={v}")
PY

say "6/8 造数据 + 训练数据体检"
if [ "$SKIP_DATA" = "1" ]; then
  echo "SKIP_DATA=1，跳过"
else
  "$PY" scripts/build_dataset.py
  "$PY" eval/split_dataset.py
fi
echo
"$PY" train/finetune_lora.py --dry-run | sed -n '1,9p' \
  || die "训练数据体检失败：多半是数据没造出来，先单独跑 scripts/build_dataset.py 看报错"

say "7/8 单元测试"
"$PY" -m unittest discover -s tests -t . 2>&1 | tail -4

say "8/8 环境自检（无卡模式）"
set +e
PYTHONIOENCODING=utf-8 "$PY" scripts/check_env.py --no-card
RC=$?
set -e
[ "$RC" -eq 0 ] || die "自检有阻塞项，按上面提示修完再重跑本脚本"

cat <<EOF

============================================================
无卡阶段完成。已经就绪的东西：
  ✓ 依赖（transformers / peft / datasets / accelerate / modelscope）
  ✓ 基座模型  $MODEL_DIR  （$(dir_gb "$MODEL_DIR") GB）
  ✓ .env      $PROJ/.env
  ✓ 数据集    1283 条训练 / 306 条测试（可用 dry-run 复核）
  ✓ 单测      全过
  ✓ 自检      无阻塞项

------------------------------------------------------------
下一步：关机 → 换成「有卡」开机（同一张卡型）→ 三行验证

  nvidia-smi
  $PY -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  $PY scripts/check_env.py

三条都对，就可以直接开训：

  mkdir -p logs
  nohup $PY train/finetune_lora.py > logs/train.log 2>&1 &
  tail -f logs/train.log

第一屏必须出现：基座模型: $MODEL_DIR
（如果显示的是 HF 的 Qwen/Qwen2.5-7B-Instruct，说明 .env 没生效，Ctrl+C 停掉，别让它联网重下）
============================================================
EOF
