#!/usr/bin/env bash
# A new version of the disease knowledge as a GitHub Release (22 November 2026), only when its content changed.
# Only the workflows of the knowledge call this (disease-review, disease-review-update): a push or a merge of code
# creates no Release. The Release is a pre-release tagged knowledge-<date>-<version>, so that it is not taken for
# a release of the software.
#   bash scripts/knowledge_release.sh <out folder of disease-review>
# Needs GH_TOKEN (the workflow's token, with contents: write).
set -euo pipefail
OUT="$1"
latest=$(gh release list --limit 200 --json tagName,createdAt \
  -q '[.[] | select(.tagName | startswith("knowledge-"))] | sort_by(.createdAt) | last | .tagName // ""')
rm -rf previous_kb && mkdir previous_kb
if [ -n "$latest" ]; then
  gh release download "$latest" -D previous_kb
  [ -f previous_kb/logs.tar.gz ] && tar xzf previous_kb/logs.tar.gz -C previous_kb
fi
python3 scripts/knowledge_release.py --previous previous_kb --run "$OUT" --release release | tee release.txt
if ! grep -q "^changed=true" release.txt; then
  echo "The knowledge did not change: no new Release (latest: ${latest:-none})."
  exit 0
fi
tag=$(sed -n 's/^tag=//p' release.txt)
tar czf release/logs.tar.gz -C release logs
assets=(release/knowledge.db release/changes.md release/version.json release/logs.tar.gz)
for f in profiles.json names.json tests.json scores.json curves.json measures.jsonl reviews.json thesaurus.jsonl; do
  [ -f "release/$f" ] && assets+=("release/$f")
done
gh release create "$tag" --prerelease --title "Disease knowledge $tag" --notes-file release/changes.md "${assets[@]}"
echo "Released $tag (previous: ${latest:-none})."
