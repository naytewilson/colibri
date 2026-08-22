#!/bin/bash
# COUNTERFACTUAL REPLAY fixtures — V4R.2
#
# Test-only simulator controls used here (never part of ordinary simulation):
#   --fixture-force-candidate-resident L:E   C path sees the candidate resident
#   --fixture-force-candidate-absent  L:E    C path skips the residency gate
#   --fixture-prequeue                L:E    is_queued pre-seeded for candidate
#
# V4R.1 receipt correction: the previous ord.req "ordinal identity" fixtures
# were INVALID_FIXTURE / VACUOUS_PROOF — demand requests preceded the
# candidate events, making the candidates resident in BOTH runs, and the old
# --assume-resident knob had contradictory semantics (forced ABSENT on the R
# path but RESIDENT on the C path). A proof-of-invariance test must first
# prove that its compared executions actually diverged on the intended policy
# outcome. The tests below use SINGLE-CANDIDATE streams and assert DIVERGENT
# OUTCOME COUNTERS BEFORE asserting ordinal identity.
#
# Usage: policy_sim_fixtures.sh <path-to-qwen36_policy_sim>
set -u
SIM="${1:?usage: $0 <qwen36_policy_sim>}"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
pass=0; failn=0
ok()  { echo "PASS $1"; pass=$((pass+1)); }
bad() { echo "FAIL $1"; failn=$((failn+1)); }

# ---------------------------------------------------------------- F1 + F2 ---
cat > "$T/f12.req" <<F12EOF
1 # qwen36_req_stream v4 fixture
2 E 0 10 3 1376256
3 E 0 11 3 1376256
4 E 0 99 3 1376256
5 E 0 98 3 1376256
6 R DEMAND -1 0 10 0.5
7 C PC -1 0 10 0 0 0.31 3 1376256
8 C PC -1 0 11 1 1 0.22 3 1376256
9 R DEMAND -1 0 99 0.1
10 R DEMAND -1 0 98 0.1
F12EOF
OUT="$($SIM "$T/f12.req" --cap 4 --svc-k 2)"
echo "$OUT" | grep -q "candidate intents=2" && ok "F1.intent-count" || bad "F1.intent-count"
echo "$OUT" | grep -q "enqueue=1" && ok "F1.sim-owns-enqueue-decision" || bad "F1.sim-owns-enqueue-decision"
PILOT_INS=$(echo "$OUT" | grep 'drained totals' | grep -o 'pilot=[0-9]*' | cut -d= -f2 | head -1)
[ "$PILOT_INS" = "1" ] && ok "F1.pilot-admission-created" || bad "F1.pilot-admission-created($PILOT_INS)"

# ---------------------------------------------------------------- F3 + F4 ---
cat > "$T/f34.req" <<F34EOF
1 # qwen36_req_stream v4 fixture
2 E 5 200 4 1769472
3 R DEMAND 0 5 200 0.9
4 R DEMAND 1 5 200 0.9
F34EOF
OUT="$($SIM "$T/f34.req" --cap 4 --svc-k 100)"
echo "$OUT" | grep -q "misses=1" && ok "F3.derived-miss-no-baseline-P" || bad "F3.derived-miss"
echo "$OUT" | grep -q "hits=1" && ok "F4.derived-hit-from-sim-residency" || bad "F4.derived-hit"
echo "$OUT" | grep -q "demand=1" && ok "F3.admission-completed" || bad "F3.admission-completed"

# -------------------------------------------------------------------- F5 ----
cat > "$T/f5.req" <<F5EOF
1 # qwen36_req_stream v4 fixture
2 E 0 1 3 100
3 E 0 2 3 100
4 E 0 3 3 100
5 E 0 4 3 100
6 E 0 5 3 100
7 E 0 6 3 100
8 E 0 7 3 100
9 E 0 8 3 100
10 R DEMAND -1 0 5 0.5
F5EOF
OUT="$($SIM "$T/f5.req" --cap 4 --svc-k 100 --check-meta)"
echo "$OUT" | grep -q "meta records consumed: 8" && ok "F5.one-digit-seq-E-consumed" || bad "F5.one-digit-seq-E-consumed"

# ------------------------------------------------------- R1 test A/B --------
# SINGLE-candidate intent stream: NO demand ever touches e41 before its
# candidate event, so e41 is ABSENT at intent time in an unmodified run.
cat > "$T/ordA.req" <<ORDEOF
1 # qwen36_req_stream v4 fixture
2 E 0 40 3 1376256
3 E 0 41 3 1376256
4 R DEMAND 0 0 40 0.5
5 C PC 0 0 41 0 0 0.60 3 1376256
6 R DEMAND 1 0 40 0.3
ORDEOF

# TEST A — resident vs absent (candidate e41):
#   A1 forces e41 RESIDENT -> DROP_RESIDENT, no enqueue
#   A2 forces e41 ABSENT   -> enqueue
"$SIM" "$T/ordA.req" --cap 4 --svc-k 3 --fixture-force-candidate-resident 0:41 --dump-ordinal "$T/ordA.txt" > "$T/outA1.txt"
"$SIM" "$T/ordA.req" --cap 4 --svc-k 3 --fixture-force-candidate-absent  0:41 --dump-ordinal "$T/ordA2.txt" > "$T/outA2.txt"
D_A1=$(grep -o 'drop(resident)=[0-9]*' "$T/outA1.txt")
D_A2=$(grep -o 'drop(resident)=[0-9]*' "$T/outA2.txt")
E_A1=$(grep -o 'enqueue=[0-9]*'      "$T/outA1.txt" | head -1)
E_A2=$(grep -o 'enqueue=[0-9]*'      "$T/outA2.txt" | head -1)
if [ "$D_A1" = "drop(resident)=1" ] && [ "$D_A2" = "drop(resident)=0" ]; then
  ok "R1A.divergent-drop-resident-counters"
else bad "R1A.divergent-drop-resident-counters ($D_A1 vs $D_A2)"; fi
if [ "$E_A1" = "enqueue=0" ] && [ "$E_A2" = "enqueue=1" ]; then
  ok "R1A.divergent-enqueue-counters"
else bad "R1A.divergent-enqueue-counters ($E_A1 vs $E_A2)"; fi
if diff -q "$T/ordA.txt" "$T/ordA2.txt" >/dev/null; then
  ok "R1A.ordinal-dump-identical-through-input-window"
else bad "R1A.ordinal-dump-identical-through-input-window"; fi

# TEST B — queued vs not queued (candidate e42), own single-candidate stream:
cat > "$T/ordB.req" <<ORDBEOF
1 # qwen36_req_stream v4 fixture
2 E 0 40 3 1376256
3 E 0 42 3 1376256
4 R DEMAND 0 0 40 0.5
5 C PC 0 0 42 0 0 0.55 3 1376256
6 R DEMAND 1 0 40 0.3
ORDBEOF
"$SIM" "$T/ordB.req" --cap 4 --svc-k 1000 --fixture-prequeue 0:42 --dump-ordinal "$T/ordB1.txt" > "$T/outB1.txt"
"$SIM" "$T/ordB.req" --cap 4 --svc-k 1000 --dump-ordinal "$T/ordB2.txt" > "$T/outB2.txt"
Q_B1=$(grep -o 'drop(queued)=[0-9]*' "$T/outB1.txt")
Q_B2=$(grep -o 'drop(queued)=[0-9]*' "$T/outB2.txt")
if [ "$Q_B1" = "drop(queued)=1" ] && [ "$Q_B2" = "drop(queued)=0" ]; then
  ok "R1B.divergent-drop-queued-counters"
else bad "R1B.divergent-drop-queued-counters ($Q_B1 vs $Q_B2)"; fi
if diff -q "$T/ordB1.txt" "$T/ordB2.txt" >/dev/null; then
  ok "R1B.ordinal-dump-identical-through-input-window"
else bad "R1B.ordinal-dump-identical-through-input-window"; fi
grep -q "service_opportunities=" "$T/ordA.txt" && ok "R1.ordinal-diagnostics-present" || bad "R1.ordinal-diagnostics-present"

# ------------------------------------------------------------- R3 EOF -------
# SINGLE-candidate stream, svc_k huge so nothing is ever serviced:
#   state A: e41 suppressed (resident) -> queue EMPTY at EOF
#   state B: e41 enqueued              -> queue NON-EMPTY at EOF
# Scored metrics freeze at EOF; cleanup drain runs afterwards and must not
# change them.
"$SIM" "$T/ordA.req" --cap 4 --svc-k 100000 --fixture-force-candidate-resident 0:41 > "$T/eofA.txt"
"$SIM" "$T/ordA.req" --cap 4 --svc-k 100000 > "$T/eofB.txt"
if grep -q 'eof_queue_depth=0' "$T/eofA.txt"; then ok "R3.eof-A-queue-empty"; else bad "R3.eof-A-queue-empty"; fi
if grep -q 'eof_queue_depth=1' "$T/eofB.txt"; then ok "R3.eof-B-queue-nonempty"; else bad "R3.eof-B-queue-nonempty"; fi
PA=$(grep 'eof demand hits' "$T/eofA.txt" | grep -o 'pilot=[0-9]*' | cut -d= -f2)
PB=$(grep 'eof demand hits' "$T/eofB.txt" | grep -o 'pilot=[0-9]*' | cut -d= -f2)
[ "$PA" = "0" ] && [ "$PB" = "0" ] && ok "R3.eof-scored-pilot-frozen-at-zero" || bad "R3.eof-scored-pilot-frozen-at-zero ($PA/$PB)"
PDB=$(grep -o 'drained totals.*pilot=[0-9]*' "$T/eofB.txt" | sed 's/.*pilot=//' | head -1)
PDA=$(grep -o 'drained totals.*pilot=[0-9]*' "$T/eofA.txt" | sed 's/.*pilot=//' | head -1)
[ "$PDB" = "1" ] && [ "$PDA" = "0" ] && ok "R3.cleanup-drain-audit-only" || bad "R3.cleanup-drain-audit-only ($PDA | $PDB)"
"$SIM" "$T/ordA.req" --cap 4 --svc-k 100000 --dump-ordinal "$T/eofOrdA.txt" --fixture-force-candidate-resident 0:41 >/dev/null
"$SIM" "$T/ordA.req" --cap 4 --svc-k 100000 --dump-ordinal "$T/eofOrdB.txt" >/dev/null
diff -q "$T/eofOrdA.txt" "$T/eofOrdB.txt" >/dev/null && ok "R3.scored-ordinal-series-identical" || bad "R3.scored-ordinal-series-identical"

# ------------------------------------------------------------- R2 tests ----
cat > "$T/g.req" <<GE
1 # qwen36_req_stream v4 fixture
2 E 5 200 4 1769472
3 R DEMAND 0 5 200 0.9
4 R DEMAND 1 5 200 0.9
GE
cat > "$T/g.oracle" <<GO
1	DEMAND	INSERT	0	5	200	4	1769472	3.000	-1	0
2	DEMAND	HIT	1	5	200	4	0	0.000	-1	0
GO
"$SIM" "$T/g.req" --oracle "$T/g.oracle" --cap 4 --svc-k 100 > "$T/g.pass" 2>&1
[ $? = 0 ] && grep -q "CURRENT_POLICY_ORACLE_GATE_PASS" "$T/g.pass" && ok "R2.gate-pass-exit0" || bad "R2.gate-pass-exit0"
sed 's/1769472/100000/' "$T/g.oracle" > "$T/g.bad"
"$SIM" "$T/g.req" --oracle "$T/g.bad" --cap 4 --svc-k 100 > "$T/g.fail" 2>&1
[ $? = 1 ] && grep -q "CURRENT_POLICY_ORACLE_GATE_FAIL" "$T/g.fail" && ok "R2.gate-fails-corrupted-oracle" || bad "R2.gate-fails-corrupted-oracle"
grep -v '^2 E' "$T/g.req" > "$T/g.nometa"
"$SIM" "$T/g.nometa" --oracle "$T/g.oracle" --cap 4 --svc-k 100 >/dev/null 2>&1
[ $? = 2 ] && ok "R2.missing-meta-fails-closed-exit2" || bad "R2.missing-meta-fails-closed-exit2"
grep -q "unresolved_waiters=0 unresolved_deferred=0" "$T/g.pass" && ok "R2.unresolved-accounting-reported" || bad "R2.unresolved-accounting-reported"

echo "SUMMARY pass=$pass fail=$failn"
[ "$failn" = "0" ]
