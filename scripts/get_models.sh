#!/usr/bin/env bash
# KAtts 模型下载脚本
# 用法：bash scripts/get_models.sh
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RELEASE_URL="${KATTS_RELEASE_URL:-}"
if [ -z "$RELEASE_URL" ]; then
  echo "请先设置模型下载地址："
  echo "  export KATTS_RELEASE_URL=https://github.com/<you>/KAtts/releases/download/v1.0"
  exit 1
fi

FILES=(
  "bert_int8.onnx"
  "s1_prefill_mask18.onnx"
  "s1_prefill_mask18.onnx.data"
  "s1_step_int8.onnx"
  "s2_pre.onnx"
  "s1_params.npz"
  "segs.tar.gz"
)

mkdir -p models segs

for f in "${FILES[@]}"; do
  case "$f" in
    segs.tar.gz) dest="." ;;
    *)           dest="models" ;;
  esac
  if [ -s "$dest/$f" ] || { [ "$f" = "segs.tar.gz" ] && [ -d segs ] && [ -n "$(ls -A segs 2>/dev/null)" ]; }; then
    echo "[skip] $f"
    continue
  fi
  echo "[get ] $f"
  curl -L --fail --retry 2 -o "$dest/$f" "$RELEASE_URL/$f"
done

if [ -f segs.tar.gz ]; then
  tar -xzf segs.tar.gz && rm -f segs.tar.gz
  echo "[ok  ] segs/ 解压完成"
fi

echo
echo "检查："
ls -lh models/ segs/ 2>/dev/null | head -30