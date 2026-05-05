#!/bin/bash

COMMIT_MSG="${1:-$(date +"%Y-%m-%d %H:%M:%S")}"
REMOTE="${2:-origin}"
BRANCH="${3:-main}"

git pull "$REMOTE" "$BRANCH"
git status
git add -A
git commit -m "$COMMIT_MSG" || echo "No changes to commit"
git push -u "$REMOTE" "$BRANCH"