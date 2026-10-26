#!/usr/bin/env bash
# 云端装环境（Linux 云镜像里执行）。**不需要 GPU**，无卡模式下也能跑完。
#
#   bash scripts/cloud_setup.sh                 # 建 .venv（继承镜像 torch）+ 清华镜像
#   bash scripts/cloud_setup.sh --no-card       # 同上，并把「没有 GPU」当成预期状态
#   bash scripts/cloud_setup.sh --no-venv       # 直接用镜像自带 python（不推荐，见下）
#   bash scripts/cloud_setup.sh --no-check      # 只装环境，不跑自检
#   VENV_DIR=/root/autodl-tmp/venv bash scripts/cloud_setup.sh   # venv 放数据盘
#
# 两个关键设计（都踩过坑，别改）：
#  1) 不安装/不升级 torch。云镜像已带 CUDA 版 torch，重装要拉几 G，
#     还容易把 CUDA 版本和驱动搞不匹配（表现为 cuda.is_available() 变 False）。
#  2) venv 用 --system-site-packages 创建。否则 venv 把镜像的 torch 隔离掉，
#     pip 会把 torch 当缺失依赖重新装一个（很可能装成 CPU 版），训练时才报 no CUDA。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MIRROR="${MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
VENV_DIR="${VENV_DIR:-.venv}"
USE_VENV=1
DO_CHECK=1
CHECK_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --no-venv)  USE_VENV=0 ;;
    --no-card)  CHECK_ARGS+=(--no-card) ;;
    --no-check) DO_CHECK=0 ;;
    --help|-h)  sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg（可用：--no-venv --no-card --no-check）"; exit 2 ;;
  esac
done

echo "== 1/6 检查 Python =="
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "找不到 $PY，先确认镜像里有 python3"; exit 1; }
"$PY" -V
echo "解释器：$(command -v "$PY")"

echo
echo "== 2/6 准备虚拟环境 =="
if [ "$USE_VENV" = "1" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    # --system-site-packages：让 venv 能看见镜像自带的 torch
    "$PY" -m venv --system-site-packages "$VENV_DIR"
    echo "已创建 $VENV_DIR（--system-site-packages）"
  else
    echo "复用已有 $VENV_DIR"
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  PY="python"
  echo "已激活：$(command -v python)"
else
  echo "跳过 venv，使用镜像自带解释器（注意：换镜像/换卡型后可能要重装）"
fi

echo
echo "== 3/6 确认 torch 可见（只检查，不安装）=="
if "$PY" -c "import torch" 2>/dev/null; then
  "$PY" - <<'PY'
import torch
print(f"torch {torch.__version__} | CUDA 编译版本 {torch.version.cuda or '无'} "
      f"| cuda.is_available={torch.cuda.is_available()}")
print("（无卡模式下 is_available=False 是正常的）")
PY
else
  cat <<'EOF'
[失败] 当前解释器里 import torch 失败。

最常见原因：venv 创建时没加 --system-site-packages，把镜像自带的 torch 隔离了。
修复：
    rm -rf .venv && bash scripts/cloud_setup.sh

另一种可能：镜像本身不带 torch（选了 Miniconda / 纯净 Ubuntu 镜像）。
此时需要装 CUDA 版（按驱动选 cu121/cu124）：
    pip install torch --index-url https://download.pytorch.org/whl/cu121
（镜像已有 torch 就别动，重装很容易把 CUDA 搞坏）
EOF
  exit 1
fi

echo
echo "== 4/6 升级 pip =="
"$PY" -m pip install -U pip -i "$MIRROR" --quiet

echo
echo "== 5/6 安装上层依赖（跳过 torch）=="
"$PY" -m pip install -i "$MIRROR" \
  "transformers>=4.44" "peft>=0.12" "datasets>=2.20" "accelerate>=0.33" \
  sentencepiece python-dotenv openai modelscope \
  || { echo "pip 安装失败：换镜像重试，例如 MIRROR=https://mirrors.aliyun.com/pypi/simple"; exit 1; }

echo
echo "== 6/6 导入验证 =="
"$PY" - <<'PY'
import importlib
for m in ("transformers", "peft", "datasets", "accelerate", "sentencepiece", "modelscope"):
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:<14} {getattr(mod, '__version__', '?')}")
    except Exception as exc:
        print(f"  失败 {m:<14} {exc}")
        raise SystemExit(1)
PY

if [ "$DO_CHECK" = "1" ]; then
  echo
  echo "== 环境自检 =="
  export PYTHONIOENCODING=utf-8
  if [ ${#CHECK_ARGS[@]} -gt 0 ]; then
    "$PY" scripts/check_env.py "${CHECK_ARGS[@]}" || true
  else
    "$PY" scripts/check_env.py || true
  fi
fi

VENV_PY="$(command -v python)"
cat <<EOF

------------------------------------------------------------
环境装好了。后面所有命令都用这个解释器：$VENV_PY
（或者先  source $VENV_DIR/bin/activate  ，然后直接用 python）

不需要 GPU 就能做的（无卡模式也行）：
  1) 下模型（15.2 GB，20-40 分钟）：
       $VENV_DIR/bin/modelscope download \\
         --model Qwen/Qwen2.5-7B-Instruct \\
         --local_dir /root/autodl-tmp/Qwen2.5-7B-Instruct
  2) 配 .env：
       cp .env.example .env && vi .env
       # FTKB_BACKEND=local
       # FTKB_BASE_MODEL_PATH=/root/autodl-tmp/Qwen2.5-7B-Instruct
  3) 造数据 + 训练数据体检（纯 CPU，30 秒）：
       $VENV_PY scripts/build_dataset.py
       $VENV_PY eval/split_dataset.py
       $VENV_PY train/finetune_lora.py --dry-run     # 期望「样本数: 1283」

必须等有卡才能做的：正式训练、真实模型评测。
完整步骤：docs/无卡模式先备环境.md
------------------------------------------------------------
EOF
