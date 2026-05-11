#!/bin/bash

COMMIT_MSG="${1:-$(date +"%Y-%m-%d %H:%M:%S")}"
REMOTE="${2:-origin}"
BRANCH="${3:-arch-v4}"

git fetch
git pull "$REMOTE" "$BRANCH"

echo "Checking FloorSet submodule..."
cd FloorSet || exit
git add -A
git commit -m "$COMMIT_MSG (submodule)" || echo "No changes in FloorSet to commit"
cd ..

git status
git add -A
git commit -m "$COMMIT_MSG" || echo "No changes to commit in main repo"
git push --recurse-submodules=no -u "$REMOTE" "$BRANCH"