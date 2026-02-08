#!/bin/bash
# 从 Latte 仓库软链共享权重到 XDiMo/ckpts/shared，便于从头训练
# 在 XDiMo 仓库根目录执行: bash scripts/setup_ckpts.sh

set -e
XDM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LATTE_ROOT="$(dirname "$XDM_ROOT")/Latte"
SHARED="$XDM_ROOT/ckpts/shared"

mkdir -p "$SHARED"

if [ ! -d "$LATTE_ROOT/share_ckpts" ]; then
    echo "未找到 Latte 仓库或 share_ckpts: $LATTE_ROOT/share_ckpts"
    echo "请先将 VAE 等权重放到 $SHARED（如 sd-vae-ft-ema 目录），或修改 LATTE_ROOT 后重试。"
    exit 1
fi

for name in sd-vae-ft-ema; do
    if [ -d "$LATTE_ROOT/share_ckpts/$name" ] && [ ! -e "$SHARED/$name" ]; then
        ln -sf "$LATTE_ROOT/share_ckpts/$name" "$SHARED/$name"
        echo "已软链: $SHARED/$name -> $LATTE_ROOT/share_ckpts/$name"
    fi
done
for name in i3d_torchscript.pt; do
    if [ -f "$LATTE_ROOT/share_ckpts/$name" ] && [ ! -e "$SHARED/$name" ]; then
        ln -sf "$LATTE_ROOT/share_ckpts/$name" "$SHARED/$name"
        echo "已软链: $SHARED/$name -> $LATTE_ROOT/share_ckpts/$name"
    fi
done

echo "完成。检查: ls -la $SHARED"
