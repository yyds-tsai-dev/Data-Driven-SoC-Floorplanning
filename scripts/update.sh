#!/bin/bash

COMMIT_MSG="${1:-$(date +"%Y-%m-%d %H:%M:%S")}"
REMOTE="${2:-origin}"
BRANCH="${3:-$(git branch --show-current)}"

git ls-remote --exit-code --heads "$REMOTE" "$BRANCH" >/dev/null
REMOTE_BRANCH_STATUS=$?

if [ "$REMOTE_BRANCH_STATUS" -eq 0 ]; then
  git pull --rebase "$REMOTE" "$BRANCH"
elif [ "$REMOTE_BRANCH_STATUS" -eq 2 ]; then
  echo "Remote branch $REMOTE/$BRANCH does not exist yet; skipping pull."
else
  echo "Unable to check remote branch $REMOTE/$BRANCH."
  exit "$REMOTE_BRANCH_STATUS"
fi

git pull "$REMOTE" "$BRANCH"
git status
git add -A
git commit -m "$COMMIT_MSG" || echo "No changes to commit"
git push -u "$REMOTE" "$BRANCH"
cp scripts/iccad2026_evaluate.py FloorSet/iccad2026contest/iccad2026_evaluate.py
