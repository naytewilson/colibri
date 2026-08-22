#!/bin/bash
# CACHE POLICY TOURNAMENT P1 — offline screening on the v4R request stream.
# Policies: P0 uniform-LRU (reference), P1 pinned-hot+LRU, P2 layer-adaptive,
# P3 segmented LRU, P4 LFU-lite. Cadence envelope svc_k={3,4,5,6}. All metrics
# are EOF-scored (policy-independent boundary); drained view is audit-only.
# Baseline deltas are computed against P0 AT THE SAME CADENCE.
#
# Usage: policy_tournament_p1.sh <qwen36_policy_sim> <requests_v4.tsv> [oracle]
set -u
SIM="${1:?usage: $0 <sim> <requests.v4> [oracle.v3]}"
REQ="${2:?missing requests stream}"
ORACLE=""
[ $# -ge 3 ] && ORACLE="$3"

CADS="3 4 5 6"
declare -A B0H B0M B0A B0B B0E B0Q   # keyed "cad:policy"

extract_eof() {  # $1=file -> sets H M DA PA DE PE DB PB Q IN W DF TRB
  local f="$1" line el bl
  line=$(grep 'eof demand hits' "$f")
  H=$(printf '%s' "$line" | grep -o 'hits=[0-9]*'   | cut -d= -f2)
  M=$(printf '%s' "$line" | grep -o 'misses=[0-9]*' | cut -d= -f2)
  DA=$(printf '%s' "$line" | grep -o 'demand=[0-9]*' | head -1 | cut -d= -f2)
  PA=$(printf '%s' "$line" | grep -o 'pilot=[0-9]*'  | head -1 | cut -d= -f2)
  el=$(grep '^eof evictions' "$f")
  DE=$(printf '%s' "$el" | grep -o 'demand=[0-9]*' | head -1 | cut -d= -f2)
  PE=$(printf '%s' "$el" | grep -o 'pilot=[0-9]*'  | tail -1 | cut -d= -f2)
  bl=$(grep 'eof evictions' "$f" | grep -o 'demand=[0-9]* pilot=[0-9]*')
  DB=$(printf '%s' "$bl" | grep -o 'demand=[0-9]*' | cut -d= -f2)
  PB=$(printf '%s' "$bl" | grep -o 'pilot=[0-9]*'  | cut -d= -f2)
  Q=$(grep -o 'eof_queue_depth=[0-9]*'     "$f" | cut -d= -f2)
  IN=$(grep -o 'eof_pilot_inflight=[0-9]*' "$f" | cut -d= -f2)
  W=$(grep -o 'eof_waiters=[0-9]*'         "$f" | cut -d= -f2)
  DF=$(grep -o 'eof_deferred=[0-9]*'       "$f" | cut -d= -f2)
  TRB=$(grep -o 'total_resident_bytes=[0-9]*' "$f" | cut -d= -f2)
}

printf '%-5s %-22s %2s %8s %8s %8s %12s %6s %6s %6s %4s %4s %4s %4s  %-12s %s\n' \
  POL PARAMS K HITS MISS D_ADM D_BYTES D_EV P_ADM P_EV Q INF W DEF dm_miss% VERDICT

for POLSPEC in \
  "P0|" \
  "P1a|--policy P1 --pin-frac 0.05|" \
  "P1b|--policy P1 --pin-frac 0.10|" \
  "P1c|--policy P1 --pin-frac 0.20|" \
  "P1d|--policy P1 --pin-frac 0.30|" \
  "P1e|--policy P1 --pin-frac 0.40|" \
  "P2|--policy P2 --rebalance-every 3200|" \
  "P3a|--policy P3 --seg-ratio 0.50|" \
  "P3b|--policy P3 --seg-ratio 0.75|" \
  "P4|--policy P4 --lfu-decay 4096|" ; do
  NAME="${POLSPEC%%|*}"
  ARGS="${POLSPEC#*|}"
  PARAMS="${ARGS#--policy }"
  for K in $CADS; do
    T="$(mktemp -d)"
    if [ "$NAME" = "P0" ]; then
      "$SIM" "$REQ" --cap 128 --svc-k "$K" ${ORACLE:+--oracle "$ORACLE"} > "$T/r.txt" 2>&1
    else
      # shellcheck disable=SC2086
      "$SIM" "$REQ" --cap 128 --svc-k "$K" $ARGS ${ORACLE:+--oracle "$ORACLE"} > "$T/r.txt" 2>&1
    fi
    extract_eof "$T/r.txt"
    KEY="$K:$NAME"
    B0H[$KEY]=$H; B0M[$KEY]=$M; B0A[$KEY]=$DA; B0B[$KEY]=$DB; B0E[$KEY]=$DE; B0Q[$KEY]=$Q

    if [ "$NAME" = "P0" ]; then
      DMP="ref"; BK=0; VER="BASELINE"
    else
      BK_KEY="$K:P0"
      BH=${B0H[$BK_KEY]:-0}; BM=${B0M[$BK_KEY]:-0}
      BA=${B0A[$BK_KEY]:-0}; BB=${B0B[$BK_KEY]:-0}; BE=${B0E[$BK_KEY]:-0}; BQ=${B0Q[$BK_KEY]:-0}
      DMP=$(awk -v m="$M" -v b="$BM" 'BEGIN{if(b>0) printf "%.2f", (b-m)*100/b; else print "n/a"}')
      BK=$(( (Q + IN + W + DF) - BQ ))
      ADM=$(awk -v d="$DMP" 'BEGIN{print (d<0?-d:d)}')
      # Adjudication terminology (V4R.1 review): the simulator is deterministic
      # — tiny effects are NOT "noise"; label them by materiality instead.
      # Materiality rule (campaign): >=20% demand-miss reduction required for
      # READY_FOR_LIVE_AB candidacy; 0.05-20% = negligible/below-threshold;
      # <=-1% = loss. There is no stochastic noise floor in this harness.
      if   $(awk -v d="$DMP" 'BEGIN{exit !(d>=20)}'); then VER="WIN@cad"
      elif $(awk -v d="$DMP" 'BEGIN{exit !(d<=-1)}'); then VER="LOSS@cad"
      elif $(awk -v d="$DMP" 'BEGIN{exit !(d>=0.5)}'); then VER="below-materiality@cad"
      elif $(awk -v d="$DMP" 'BEGIN{exit !(d>=0.05)}'); then VER="negligible@cad"; else VER="null@cad"; fi
      # backlog-shift law
      if [ "$BK" -gt 64 ]; then VER="$VER+BACKLOG_SHIFT"; fi
    fi
    printf '%-5s %-22s %2s %8s %8s %8s %12s %6s %6s %6s %4s %4s %4s %4s  %-12s %s\n' \
      "$NAME" "$PARAMS" "$K" "$H" "$M" "$DA" "$DB" "$DE" "$PA" "$PE" "$Q" "$IN" "$W" "$DF" "dm=$DMP%" "$VER"
    rm -rf "$T"
  done
done
