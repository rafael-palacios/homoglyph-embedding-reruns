#!/bin/bash
# Writes the round-4 job lists, one per environment, so that the two machines never share a list:
#   $WORK/jobs/r4_small.txt  SmallGPU (4 x 16 GB, 64 GB RAM): evaluations first, then short training runs
#   $WORK/jobs/r4_large.txt  LargeGPU (1 x 71 GB): the long runs (wide whole-image CNN, two-block
#                            cross-attention at 8 px)
#   $WORK/jobs/r4_after.txt  run on either machine once BOTH lists have finished: scores the new
#                            controlled-data models on the original test split, in unseen fonts and on
#                            the process names
#
# Usage (on the server, with WORK and ITAU set and the paper's Python environment active):
#   bash make/round4_jobs.sh
# Then, before starting any GPU list, build the render caches once (see the end of this script's output).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${WORK:?set WORK}" "${ITAU:?set ITAU}"
J="$WORK/jobs"
mkdir -p "$J"
MJ="python $HERE/make_jobs.py"

{
  echo "# SmallGPU, round 4. Run with: GPUS=\"0 1 2\" bash make/run_gpu.sh \$WORK/jobs/r4_small.txt"
  echo "# 1) existing controlled-data models (p0) in unseen fonts and on process names (evaluation only)"
  $MJ r4_ctrl_generalization
  echo "# 2) BiLSTM, Transformer, one-block cross-attention trained on controlled Woodbridge (27 runs, ~1 h each)"
  $MJ r4_ctrl_mechanism --models bilstm transformer cross_attention_1block
  echo "# 3) common slice width 8/4 on p0: BiLSTM, Transformer, one-block cross-attention (9 runs)"
  $MJ r4_width8 --models bilstm transformer cross_attention_1block
} > "$J/r4_small.txt"

{
  echo "# LargeGPU, round 4. Run with: bash make/run_gpu.sh \$WORK/jobs/r4_large.txt"
  echo "# 1) wide whole-image CNN trained on controlled Woodbridge (9 runs, ~8 h each)"
  $MJ r4_ctrl_mechanism --models whole_image_cnn_wide
  echo "# 2) two-block cross-attention at slice width 8/4 on p0 (3 runs)"
  $MJ r4_width8 --models cross_attention_2block
} > "$J/r4_large.txt"

{
  echo "# After both lists have finished, round 4. Run on either machine with run_gpu.sh."
  echo "# 1) new controlled-data mechanism models scored on the original Woodbridge test split"
  $MJ r4_ctrl_mechanism_orig_eval
  echo "# 2) new controlled-data mechanism models (p0) in unseen fonts and on process names"
  $MJ r4_ctrl_generalization --ctrl-sets woodbridge --models bilstm transformer cross_attention_1block whole_image_cnn_wide
} > "$J/r4_after.txt"

for f in r4_small r4_large r4_after; do
  echo "$J/$f.txt: $(grep -cv '^#' "$J/$f.txt") jobs"
done
cat <<EOF

Before starting run_gpu.sh on either machine, build the render caches ONCE (CPU only, one machine),
so the two machines never build the same cache at the same time on the shared disk:
  python $HERE/build_caches.py --data woodbridge_p0 woodbridge_ctrl_p0 woodbridge_ctrl_p1 woodbridge_ctrl_p2 \\
      synthetic_ctrl_p0 process --workers 16
(woodbridge_p1/p2 are only read in r4_after, through their test split; add them if their caches are missing.)
EOF
