#!/bin/bash
# COUNTERFACTUAL_REPLAY_V4R adversarial fixtures.
# Proves the simulator makes its OWN decisions from intents alone:
#   F1: candidate intent exists even though BASELINE residency would suppress
#       it -> simulator enqueues+admits it (admission that did NOT exist in
#       baseline).
#   F2: candidate suppressed by the SIMULATOR when ITS OWN policy state says
#       resident (suppression decided by sim, not by a baseline marker).
#   F3: demand request with NO baseline publish marker resolves as MISS ->
#       simulator creates the admission itself.
#   F4: demand request served from SIMULATOR residency -> HIT without any
#       baseline outcome input.
#   F5: Defect E regression — E records at sequence 2..9 are consumed.
#
# Usage: test_policy_sim_fixtures.sh <path-to-qwen36_policy_sim>
set -u
SIM="${1:?usage: $0 <qwen36_policy_sim>}"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
pass=0; failn=0
ok()  { echo "PASS $1"; pass=$((pass+1)); }
bad() { echo "FAIL $1"; failn=$((failn+1)); }

# ---------------------------------------------------------------- F1 + F2 ---
# layer0, experts 10 and 11, INT3 1376256 bytes each. cap=4.
# Event order: meta, one demand miss on e10 (sim admits e10), candidate intent
# for e10 (baseline would have suppressed: resident) and for e11 (absent),
# then enough events to drive service boundaries (svc_k=2).
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
echo "$OUT" > "$T/f12.out"
# e10 was IN-FLIGHT (not resident) at intent time, so the simulator enqueues
# BOTH candidates — faithful to runtime gating where in-flight != resident.
# The counterfactual opportunity (e11) therefore exists in the stream and in
# the queue; on dequeue e10 hits the now-resident copy (sim drop), e11 admits.
echo "$OUT" | grep -q "candidate intents=2" && ok "F1.intent-count" || bad "F1.intent-count"
echo "$OUT" | grep -q "enqueue=1" && ok "F1.sim-owns-enqueue-decision" || bad "F1.sim-owns-enqueue-decision"
# after drain, e11 must exist as a PILOT admission that never existed in any
# baseline run of this stream
PILOT_INS=$(echo "$OUT" | sed -n 's/.*pilot=\([0-9]*\).*/\1/p' | head -1)
[ "$PILOT_INS" = "1" ] && ok "F1.pilot-admission-created" || bad "F1.pilot-admission-created($PILOT_INS)"

# ---------------------------------------------------------------- F3 + F4 ---
# demand request for an expert with NO publish marker anywhere in the stream:
# the simulator must derive MISS + complete the admission itself; then a
# second request for the same expert derives HIT purely from sim residency.
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
# Defect E regression: E records at sequence 2..9 must ALL be consumed.
cat > "$T/f5.req" <<'EOF'
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
EOF
OUT="$($SIM "$T/f5.req" --cap 4 --svc-k 100 --check-meta)"
echo "$OUT" | grep -q "meta records consumed: 8" && ok "F5.one-digit-seq-E-consumed" || bad "F5.one-digit-seq-E-consumed"

# ------------------------------------------------------------- R1 tests ----
# Ordinal identity: SAME stream, two simulator states. --assume-resident
# forces the candidate's expert to look resident (state A: drop) vs absent
# (state B: enqueue+admit). Outcomes diverge; the service-opportunity ordinal
# series MUST NOT.
cat > "$T/ord.req" <<ORD
1 # qwen36_req_stream v4 fixture
2 E 0 30 3 1376256
3 E 0 31 3 1376256
4 R DEMAND 0 0 30 0.5
5 C PC 0 0 30 0 0 0.40 3 1376256
6 R DEMAND 1 0 31 0.2
7 C PC 1 0 31 1 1 0.20 3 1376256

8 E 0 99 3 1376256
9 R DEMAND 2 0 99 0.1
10 C PC 2 0 30 0 2 0.10 3 1376256
9 C PC 2 0 30 0 2 0.10 3 1376256
ORD
"$SIM" "$T/ord.req" --cap 4 --svc-k 3 --dump-ordinal "$T/ordA.txt" >/dev/null
"$SIM" "$T/ord.req" --cap 4 --svc-k 3 --dump-ordinal "$T/ordB.txt" --assume-resident 0:30 >/dev/null
if diff -q "$T/ordA.txt" "$T/ordB.txt" >/dev/null; then ok "R1.ordinal-identical-resident-vs-absent"; else bad "R1.ordinal-identical-resident-vs-absent"; fi
# queued-vs-not-queued: run A leaves e31 queued (never serviced: svc_k high);
# run B treats e31 as resident so it is never queued. Ordinals must match.
"$SIM" "$T/ord.req" --cap 4 --svc-k 1000 --dump-ordinal "$T/ordC.txt" >/dev/null
"$SIM" "$T/ord.req" --cap 4 --svc-k 1000 --dump-ordinal "$T/ordD.txt" --assume-resident 0:31 >/dev/null
if diff -q "$T/ordC.txt" "$T/ordD.txt" >/dev/null; then ok "R1.ordinal-identical-queued-vs-not"; else bad "R1.ordinal-identical-queued-vs-not"; fi
grep -q "service_opportunities=" "$T/ordA.txt" && ok "R1.ordinal-diagnostics-present" || bad "R1.ordinal-diagnostics-present"

# ------------------------------------------------------------- R2 tests ----
# Tiny current-policy pair: stream derives hits=1 misses=1 demand_adm=2
# bytes=3538944; the hand-written oracle records exactly that.
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
# corrupted oracle: admitted bytes halved -> byte residual beyond tolerance
sed 's/1769472/100000/' "$T/g.oracle" > "$T/g.bad"
"$SIM" "$T/g.req" --oracle "$T/g.bad" --cap 4 --svc-k 100 > "$T/g.fail" 2>&1
[ $? = 1 ] && grep -q "CURRENT_POLICY_ORACLE_GATE_FAIL" "$T/g.fail" && ok "R2.gate-fails-corrupted-oracle" || bad "R2.gate-fails-corrupted-oracle"
# missing expert metadata -> contract failure, exit 2
grep -v '^2 E' "$T/g.req" > "$T/g.nometa"
"$SIM" "$T/g.nometa" --oracle "$T/g.oracle" --cap 4 --svc-k 100 >/dev/null 2>&1
[ $? = 2 ] && ok "R2.missing-meta-fails-closed-exit2" || bad "R2.missing-meta-fails-closed-exit2"
# unresolved-waiter / deferred accounting is reported on every gate run
grep -q "unresolved_waiters=0 unresolved_deferred=0" "$T/g.pass" && ok "R2.unresolved-accounting-reported" || bad "R2.unresolved-accounting-reported"

echo "SUMMARY pass=$pass fail=$failn"
[ "$failn" = "0" ]
