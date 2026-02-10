#!/usr/bin/env bash
# 推送本仓库到超算。排除规则见 .rsync_exclude
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
EXCLUDE="${REPO}/.rsync_exclude"
[[ -f "$EXCLUDE" ]] || { echo "缺少 $EXCLUDE"; exit 1; }
FETCH="${REPO}/fetch.sh"
[[ -f "$FETCH" ]] || { echo "缺少 $FETCH"; exit 1; }

REMOTE="${REMOTE:-liuhuijie@10.150.2.4}"
REMOTE_DIR="${REMOTE_DIR:-/public/home/liuhuijie/dits/XDiMo/}"

echo "推送前先拉取超算最新内容（会触发 git 提交/推送）..."
bash "$FETCH"

sshpass -p "${SSHPASS:-Hpc1234#$}" rsync -avzP --exclude-from="$EXCLUDE" "${REPO}/" "${REMOTE}:${REMOTE_DIR}"
sshpass -p "${SSHPASS:-Hpc1234#$}" ssh -o StrictHostKeyChecking=accept-new "${REMOTE}" "rm -rf ${REMOTE_DIR}src" 2>/dev/null || true
echo "推送完成（已移除远程 src/）"
