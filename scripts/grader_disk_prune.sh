#!/usr/bin/env bash
# Periodic grader disk hygiene. SAFE BY DESIGN:
#  - removes only course-gen-outcome:* images (per-submission throwaway)
#    older than 30 min, so an in-flight grader build (completes in <=6m)
#    is never touched;
#  - dangling images, and build cache older than 30 min;
#  - never targets course-gen-learner-studio / postgres / app / DB
#    (different image refs; docker also skips images used by containers).
set -uo pipefail
LOG=/opt/course-gen-codex/logs/grader-prune.log
mkdir -p /opt/course-gen-codex/logs
exec >>"$LOG" 2>&1

echo "=== $(date -u +%FT%TZ) prune start"
df -h / | tail -1
now=$(date +%s)
for id in $(docker images --filter=reference='course-gen-outcome:*' --format '{{.ID}}' | sort -u); do
  created=$(docker inspect -f '{{.Created}}' "$id" 2>/dev/null) || continue
  cts=$(date -d "$created" +%s 2>/dev/null) || continue
  age=$(( now - cts ))
  if [ "$age" -gt 1800 ]; then
    docker rmi -f "$id" >/dev/null 2>&1 && echo "removed image $id (age $((age/60))m)"
  fi
done
docker image prune -f >/dev/null 2>&1 && echo "dangling images pruned"
docker builder prune -f --filter 'until=30m' >/dev/null 2>&1 && echo "build cache >30m pruned"
df -h / | tail -1
echo "=== done"
