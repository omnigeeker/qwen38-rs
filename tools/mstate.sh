#!/usr/bin/env bash
# Print the machine state that has been shown to swing these measurements by 2-4x, so
# that two runs can be judged comparable before their numbers are compared at all.
cd "$(dirname "$0")/.." || exit 1
printf 'state %s | %s | load %s | swap %s | gpu %s | stray_qwen38 %s\n' \
  "$(date '+%H:%M:%S')" \
  "$(pmset -g ps 2>/dev/null | head -1 | tr -s ' ')" \
  "$(sysctl -n vm.loadavg 2>/dev/null | tr -d '{}')" \
  "$(sysctl -n vm.swapusage 2>/dev/null | awk '{print $3, $6}')" \
  "$(ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1)" \
  "$(pgrep -x qwen38 2>/dev/null | wc -l | tr -d ' ')"
