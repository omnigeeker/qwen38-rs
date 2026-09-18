#!/bin/bash
URL="$1"; OUT="$2"; COOKIE="$3"; SIZE="$4"; N=8
rm -f /tmp/part_*
CHUNK=$(( SIZE / N + 1 ))
for i in $(seq 0 $((N-1))); do
  START=$(( i * CHUNK )); END=$(( START + CHUNK - 1 ))
  [ $END -ge $SIZE ] && END=$(( SIZE - 1 ))
  curl -s --max-time 600 --compressed -A "Mozilla/5.0 Chrome/126" -H "Referer: https://www.sse.com.cn/" \
    -H "Cookie: $COOKIE" -r ${START}-${END} "$URL" -o /tmp/part_$i &
done
wait
cat /tmp/part_* > "$OUT"
ls -la "$OUT"
