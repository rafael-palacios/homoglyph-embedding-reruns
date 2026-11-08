# Repeated experiments for "Embedding Models for Fine-Grained Visual Homoglyph Detection"

This repository contains the scripts that produced every result of the repeated experiments in
*Embedding Models for Fine-Grained Visual Homoglyph Detection* (Liu, Boriel, Xin, Palacios, Wanderley,
Rinaldo and Gupta). It uses the paper's model and training code without modification, and adds data
preparation, job lists, evaluation and aggregation. The aggregated results reported in the paper are in
[`results/`](results/).

| Experiment | Stage | What it produces |
|---|---|---|
| Variability over seeds and data partitions | `main` | 4 models × 2 datasets × 3 partitions × 5 seeds; mean ± s.d. |
| Sliced models vs. whole-image CNN | `main` | Paired tests on the same partition and seed (DeLong per run; t-test and Wilcoxon across runs) |
| Unseen fonts | `fonts` | Test ROC-AUC with the test split rendered in 4 fonts not used in training |
| Character-set shortcut | `shortcut_baseline.py`, `make_controlled.py`, `ctrl`, `ctrl_eval` | How far non-visual cues go; a shortcut-controlled version of the synthetic data; models tested on it and retrained on it |
| Independently collected data | `process` | Test ROC-AUC on Woodbridge's process-name test split |
| Slice-width sensitivity | `sweep` | Selected model of each dataset with slice widths 4–32 px, 3 seeds |

## The paper's code

The model and training code is in github.com/setliu22/Itau, **not on its default branch**: it is on branch
`agent-update-domains-spoof-no-com-metrics` (commit `4f93925`), in `synth_dataset/architecture_search/`.
That branch also holds the screening results from which the frozen configurations in `configs/` were
taken (`model_results/architecture_search_exports/`).

The models are trained with the paper's final protocol (`run_selected_winner_training.py`): 25 epochs, no
early stopping, the checkpoint with the best validation ROC-AUC is kept and then scored once on the test
split. **Hyperparameters are not re-tuned**; each model uses its best screening configuration. Only the
seed, the partition and, in the sweep, the slice width change.

| Model | Woodbridge config | Synthetic config |
|---|---|---|
| Interaction CNN (selected for Woodbridge in screening) | 16-px slices, stride 8 | 16-px slices, stride 16 |
| Two-block cross-attention (selected for synthetic in screening) | 6-px slices, stride 3 | 6-px slices, stride 6 |
| Whole-image CNN (unsliced reference) | — | — |
| Conv1D (sliced, no pair interaction) | 6-px slices | 6-px slices |

## Resources

The paper's runs used NVIDIA H200 GPUs partitioned into MIG instances: 18 GB instances for all runs except
the slice-width sweep, which needs a larger one (below).

| | Runs | GPU-hours measured |
|---|---|---|
| `main` | 120 | 255 |
| `ctrl` | 36 | 41 |
| `sweep` | 36 | 150 |
| `fonts`, `process`, `ctrl_eval` (evaluation only) | 260 | a few minutes each |

Training time per run on an 18 GB instance, mean over runs: Woodbridge 3.7 h (Interaction CNN), 3.9 h
(two-block cross-attention), 1.9 h (whole-image CNN), 0.8 h (Conv1D); synthetic 1.5–2.0 h. Each concurrent
job also needs about 16 GB of RAM and 5 CPU cores (4 data-loading workers). Rendered images and data take
30–40 GB of disk.

The Interaction CNN compares every slice of one name with every slice of the other, so halving the slice
width and stride quadruples its interaction map. With 4-px slices (stride 2) one Woodbridge run needs about
67 GB of GPU memory and about 12 hours; with 8-px slices it takes about 9 hours. Set this before running the
sweep, to reduce memory fragmentation:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 0. Get the code and the data

```bash
git clone https://github.com/setliu22/Itau.git
git -C Itau switch -c paper 4f93925   # the paper's code (branch agent-update-domains-spoof-no-com-metrics)
git clone https://github.com/setliu22/fine-grained-homoglyph-detection.git   # contains Woodbridge's data
git clone https://github.com/rafael-palacios/homoglyph-embedding-reruns.git make
```

Set two variables (absolute paths):

```bash
export ITAU=$PWD/Itau
export WORK=$HOME/make_results   # data, rendered images, runs and results go here
```

`fine-grained-homoglyph-detection/data/raw/` holds Woodbridge's `domains_spoof.pkl` and
`process_spoof.pkl`; the same files are in github.com/endgameinc/homoglyph.

## 1. Python environment

```bash
python3.11 -m venv ~/venvs/make && source ~/venvs/make/bin/activate
pip install -r make/requirements.txt
```

## 2. Woodbridge partitions

```bash
python make/make_partitions.py --partitions 0 1 2 \
    --domains-pkl fine-grained-homoglyph-detection/data/raw/domains_spoof.pkl \
    --process-pkl fine-grained-homoglyph-detection/data/raw/process_spoof.pkl \
    --val10k $ITAU/synth_dataset/base_datasets/validate_pairs_ref_10k.parquet
```

- **p0** is the original split of Woodbridge et al.
- **p1, p2** re-assign the reference names to train/validation/test at random, with the same number of
  names per split. In Woodbridge's data each reference has 9 spoofs and 5 negatives, and the negatives
  are the 5 reference names of the same split nearest in Levenshtein distance (ties at random; this rule
  holds for every reference checked). The script keeps each reference with its spoofs and rebuilds the
  negatives within each new split by the same rule. Simply re-splitting the original pairs would leak:
  test negatives would contain training references.
- `$WORK/data/partitions.json` reports the checks: no reference name shared between splits, 98% of test
  pairs contain no string seen in training (as in p0), and the same mean edit distance of the negatives in
  each split.

The script also writes the base files that the synthetic generator needs and the process-name test split.

## 3. Synthetic partitions

All three synthetic partitions are generated with the paper's generator, unchanged, on each partition's
base files, with generation seed 42 + k. About 1.5 hours of CPU each, one core.

The generator is run through `run_generator.py`, which changes only how it stores its table of spoof
variants: the unwrapped generator keeps edit descriptions for every variant of every name, some 72 GB for
p0. The wrapper keeps each variant once and rebuilds its edit descriptions when needed, with the
generator's own functions; peak memory is 5.5 GB for p0, and its output files are byte-for-byte identical
to an unwrapped run's (checked on p0).

```bash
mkdir -p $WORK/logs
for P in 0 1 2; do
  python make/prepare_synthetic.py --partition $P --generate > $WORK/logs/synth_p$P.log 2>&1 || { echo "p$P failed"; break; }
done
tail -2 $WORK/logs/synth_p*.log      # each log ends with "wrote ...data/synthetic_p<k>"
```

`synth_dataset/generated_datasets/mix65/generation_metrics.json` in the paper's repository records the
original generation run, and the regenerated p0 reproduces every count in it: split sizes and class counts
(976,122 / 9,999 / 256,886 pairs), real-name pools (69,723 / 3,488 / 18,349), 799,076 distinct positive
candidates and 45 negative length buckets.

Negative candidates are drawn from the reference names of the Woodbridge benchmark (91,560 names for p0),
which the generator uses by itself when no other list is given. The original run had used a filtered
subset of 87,211 of these names that was not published, so the regenerated p0 is a new draw of the same
design rather than a copy of the original file; the regenerated data can be rebuilt entirely from public
sources. (`--negative-names swap-lookup` uses the 87,337 names of the archived adjacent-swap lookup
instead; it has no names of 1 to 3 characters and gives 42 negative length buckets instead of 45.)

## 4. Shortcut control and non-visual baselines (CPU)

```bash
for P in 0 1 2; do python make/make_controlled.py --partition $P; done
python make/shortcut_baseline.py --data woodbridge_p0 synthetic_p0 synthetic_ctrl_p0
```

In the synthetic data, positive candidates carry the non-ASCII substitutions and negative candidates are
unmodified legitimate names, so the class can be read off the candidate's character set.
`make_controlled.py` writes `synthetic_ctrl_p<k>`, in which every negative candidate (a legitimate name
paired with a different reference) receives homoglyph substitutions copied from the positives, with the
same number of non-ASCII characters per candidate at each name length, the same rate of substitution per
source character and the same choice of replacements. The candidate still imitates a different name, so
the label stays 0, but its character set no longer reveals the class.
`data/synthetic_ctrl_p<k>/control_report.json` shows the non-ASCII distributions of both classes before
and after.

`shortcut_baseline.py` trains gradient-boosting classifiers that never see an image, on five feature
sets. Test ROC-AUC on p0:

| Features | Woodbridge | Synthetic | Synthetic, controlled |
|---|---|---|---|
| Candidate's non-ASCII count | 0.822 | 0.999 | 0.503 |
| Candidate's character composition | 0.873 | 0.999 | 0.597 |
| Composition of both strings | 0.888 | 0.999 | 0.682 |
| The two lengths | 0.876 | 0.915 | 0.915 |
| All of the above + edit distances | 0.990 | 1.000 | 0.999 |

The full table is in `results/shortcut_baseline.csv`.

## 5. Render caches (CPU, recommended)

```bash
python make/build_caches.py --workers 32 --data woodbridge_p0 woodbridge_p1 woodbridge_p2 \
    synthetic_p0 synthetic_p1 synthetic_p2 synthetic_ctrl_p0 synthetic_ctrl_p1 synthetic_ctrl_p2
```

As in the paper, every name is rendered once (DejaVu Sans, 32 px, white on black) into
`$WORK/render_cache/`, and the cache is checked against fresh renders. Training builds a missing cache
itself, but then the GPU waits.

## 6. Check that the pipeline reproduces the paper

```bash
mkdir -p $WORK/jobs
python make/make_jobs.py main --partitions 0 --seeds 7 > $WORK/jobs/check.txt
bash make/run_gpu.sh $WORK/jobs/check.txt
grep "test ROC-AUC" $WORK/logs/check/*.log
```

The Woodbridge Interaction CNN with the original seed (7) on p0 reached a test ROC-AUC of 0.99432 (best
epoch 11), against 0.9942 (best epoch 10) in the original run; cuDNN is deterministic, but a different GPU
or library version changes the last digits. The synthetic p0 is regenerated (step 3), so two-block
cross-attention comes close to the original 0.99999 (0.999989) without matching it exactly.

## 7. Run the stages (GPU)

```bash
python make/make_jobs.py main     --partitions 0 1 2 --seeds 7 1 2 3 4 > $WORK/jobs/main.txt
python make/make_jobs.py sweep    --seeds 7 1 2                        > $WORK/jobs/sweep.txt
python make/make_jobs.py ctrl     --partitions 0 1 2 --seeds 7 1 2     > $WORK/jobs/ctrl.txt
bash make/run_gpu.sh $WORK/jobs/main.txt      # then sweep.txt and ctrl.txt
```

After `main` has finished:

```bash
python make/make_jobs.py fonts     --seeds 7 1 2 3 4                    > $WORK/jobs/fonts.txt
python make/make_jobs.py process   --seeds 7 1 2 3 4                    > $WORK/jobs/process.txt
python make/make_jobs.py ctrl_eval --partitions 0 1 2 --seeds 7 1 2 3 4 > $WORK/jobs/ctrl_eval.txt
bash make/run_gpu.sh $WORK/jobs/fonts.txt     # then process.txt and ctrl_eval.txt
```

- `run_gpu.sh` uses the devices in `CUDA_VISIBLE_DEVICES` (GPU indices or MIG UUIDs), otherwise all GPUs.
  `GPUS="0 1 2 3"` restricts them; `JOBS_PER_GPU=2` runs two jobs per GPU if memory, CPU and RAM allow.
- Run each job list in **one** session only: two sessions running the same list duplicate its jobs.
- The three 4-px Interaction CNN runs of the sweep need about 70 GB of GPU memory (the paper used a 71 GB
  MIG instance with `expandable_segments`, one job at a time). Putting them in their own job list, run on
  that GPU, and the rest of the sweep in another avoids out-of-memory errors.
- Every job skips finished work, so a job list can simply be started again. Resuming an interrupted
  training from its checkpoint fails with some PyTorch versions (`RNG state must be a torch.ByteTensor`);
  delete the run folder of an interrupted training so that it starts again from scratch.
- On a SLURM cluster, submit each line of a job list as its own job instead.
- The `fonts` stage uses DejaVu Serif, DejaVu Sans Mono and STIX General (shipped with matplotlib) and
  GNU Unifont (in the paper's repository). Other font files can be given with `--fonts a.ttf b.ttf ...`.
  Characters missing from a font are drawn as boxes, so the results are also given for the pairs that a
  font fully covers.

Failed jobs of a job list:

```bash
grep -L -E 'test ROC-AUC|ROC-AUC [0-9]|already (finished|evaluated)' $WORK/logs/<job list>/job*.log
```

## 8. Results

```bash
python make/aggregate.py
```

Writes to `$WORK/results/`. The files in [`results/`](results/) are the output of this step for the
paper.

| File | Contents |
|---|---|
| `main_summary.csv`, `main_table.tex` | Test metrics, mean ± s.d. over 15 runs per model and dataset |
| `by_partition.csv` | The same per partition, to separate data and seed variability |
| `paired_tests.csv`, `delong_per_run.csv` | Paired comparisons between models on the same partition and seed |
| `fonts.csv` | Unseen fonts |
| `shortcut_baseline.csv`, `controlled.csv` | Non-visual baselines; models on the shortcut-controlled data |
| `process.csv` | Process-name test split |
| `slice_sweep.csv`, `slice_sweep.pdf` | Slice-width sensitivity |
| `det_curves.pdf` | DET curves (p0, seed 1) |
| `report.txt`, `runs.csv` | Everything above as text; one row per run and evaluation |

## Files

| File | Purpose |
|---|---|
| `configs/*.yaml` | Frozen configurations of the 4 models on each dataset, from the paper's screening |
| `common.py` | Imports the paper's code, replaces its hard-coded paths, relaxes the slice-width check for the sweep, evaluation |
| `make_partitions.py` | Woodbridge partitions, base files for the generator, process-name test split |
| `prepare_synthetic.py` | Synthetic partitions: runs the paper's generator |
| `run_generator.py` | Runs the paper's generator with a compact variant table (5.5 GB instead of ~72 GB) |
| `make_controlled.py` | Shortcut-controlled synthetic partitions |
| `shortcut_baseline.py` | Non-visual baselines |
| `build_caches.py` | Render caches |
| `run_one.py` | Trains one model with one seed on one partition and scores it on the test split |
| `evaluate.py` | Scores a trained model on other pairs or in another font |
| `make_jobs.py`, `run_gpu.sh` | Job lists and a queue that keeps every GPU busy |
| `aggregate.py` | Tables, statistical tests and figures |

## Notes

- The paper's code validates on the full Woodbridge validation split of 51,380 pairs
  (`architecture_search/constants.py`); 9,999 is the size of the synthetic validation split. This package
  keeps 51,380, as the original runs did.
- Woodbridge et al., *Detecting homoglyph attacks with a Siamese neural network*, IEEE Security and Privacy
  Workshops 2018. Their data: github.com/endgameinc/homoglyph.
