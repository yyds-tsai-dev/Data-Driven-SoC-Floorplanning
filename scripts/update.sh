#!/bin/bash

COMMIT_MSG="${1:-$(date +"%Y-%m-%d %H:%M:%S")}"
REMOTE="${2:-origin}"
BRANCH="${3:-$(git branch --show-current)}"

git pull "$REMOTE" "$BRANCH"
git status
git add -A
git commit -m "$COMMIT_MSG" || echo "No changes to commit"
git push -u "$REMOTE" "$BRANCH"
# cp scripts/iccad2026_evaluate.py FloorSet/iccad2026contest/iccad2026_evaluate.py
