#!/bin/bash
set -u
# 1) full tree listings (case-insensitive search for cambricon) for inference & training results repos
for kind in inference_results training_results; do
 for v in 0.7 1.0 1.1 2.0 2.1 3.0 3.1 4.0 4.1 5.0 5.1 6.0 6.1; do
  for br in main master; do
    out=$(curl -sS --max-time 60 "https://api.github.com/repos/mlcommons/${kind}_v$v/git/trees/$br?recursive=1" 2>/dev/null)
    echo "$out" | grep -q '"tree"' && { echo "$out" > "tree_${kind}_v$v.json"; echo "OK ${kind}_v$v ($br)"; break; }
  done
 done
done
# 2) full downloads of summary_results.json
for v in 5.0 5.1 6.0 6.1; do
  curl -sS -L --max-time 300 -o "sr_$v.json" "https://raw.githubusercontent.com/mlcommons/inference_results_v$v/main/summary_results.json" && echo "DL sr_$v $(wc -c < sr_$v.json)"
done
