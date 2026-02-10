#!/usr/bin/env bash
# 从超算拉取到本仓库。排除规则见 .rsync_exclude
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
EXCLUDE="${REPO}/.rsync_exclude"
[[ -f "$EXCLUDE" ]] || { echo "缺少 $EXCLUDE"; exit 1; }

REMOTE="${REMOTE:-liuhuijie@10.150.2.4}"
REMOTE_DIR="${REMOTE_DIR:-/public/home/liuhuijie/dits/XDiMo/}"

sshpass -p "${SSHPASS:-Hpc1234#$}" rsync -avzP --update --exclude-from="$EXCLUDE" "${REMOTE}:${REMOTE_DIR}" "${REPO}/"
echo "拉取完成: ${REPO}/"

read -r -p "是否提交并 push? (yes/no): " answer
answer="$(echo "$answer" | tr '[:upper:]' '[:lower:]')"
if [[ "$answer" != "yes" && "$answer" != "y" ]]; then
  echo "已跳过 push，退出。"
  exit 0
fi

read -r -p "请输入 commit 说明: " msg
if [[ -z "${msg// /}" ]]; then
  echo "未输入说明，已取消。"
  exit 1
fi

cd "$REPO"
if ! git status --porcelain | grep -q .; then
  echo "工作区无变更，跳过 commit/push。"
  exit 0
fi
git add -A
git commit -m "$msg"
# 先 rebase 远程，尽量自动解决非冲突场景（可提示确认）
branch=$(git rev-parse --abbrev-ref HEAD)
read -r -p "即将执行 git pull --rebase --autostash，是否继续？(y/n): " rebase_answer
rebase_answer="$(echo "$rebase_answer" | tr '[:upper:]' '[:lower:]')"
if [[ "$rebase_answer" != "y" && "$rebase_answer" != "yes" ]]; then
  echo "已跳过 rebase，请手动同步远程后再 push。"
  exit 1
fi
if ! git pull --rebase --autostash origin "$branch"; then
  echo "自动 rebase 失败（可能有冲突）。请手动解决后再执行 git push。"
  exit 1
fi
git push
echo "已提交并 push 完成。"
