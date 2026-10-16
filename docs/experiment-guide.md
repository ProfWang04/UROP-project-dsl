# Experiment implementation guide

This is the maintained implementation reference for the experiment. Update it in
any change that alters sampling, model inputs, optimization, scoring, defaults,
or saved artifacts. The source code is authoritative when diagnosing a mismatch.

Last updated: 2026-10-16. Current scoring contract: `evaluation_version = 2`.

## 1. Research question and scope

The experiment trains a small causal Transformer using next-token prediction
(NTP) on executable programs. It investigates how program validity and the ability
to generate withheld API compositions change with training-set size N and API
count k. An interpreter supplies deterministic verification; there is no RL,
reward optimization, constrained decoding, or verifier feedback during training.

The current hard and mixture arms fix k=6. The implementation does not yet execute
the proposed full k={4,6,8,10} grid. The toy soft arm supports varying enabled APIs.
A fixed-epoch N curve also changes the number of optimizer updates with N: it is
not a controlled comparison at fixed compute or proof of a sharp phase transition.

There are two distinct evaluation questions:

- Teacher forcing: does the model predict a structurally legal next token on an
  actual unseen IID or held-out reference program?
- Sampled completion: given an initial statement from a test program, does the
  model finish an executable program satisfying that split's constraints?

Completion is open ended. It does not require reproducing the reference suffix,
its numeric outputs, or a requested input/output specification. A successful
hold-out completion must actually contain the withheld composition. This is a
constrained program-generation diagnostic, not a SCAN-style instruction-to-program
accuracy benchmark. First-statement prompts can coincide across different test
programs and across splits; split identity is a scoring constraint, not a token
provided to the model. Use the teacher-forced hold-out score alongside completion
scores when interpreting compositional generalization.

## 2. Vocabulary and executable language

`src/dsl/tokens.py` defines the complete vocabulary and token/ID mapping. Special
tokens are PAD, BOS, EOS, and UNK. Arithmetic tokens are digits 0 through 9,
`+`, `-`, `*`, and parentheses. Variables are the eight names `a` through `h`.
Unknown input tokens encode as UNK and cannot pass the generation verifier.

L0 (`src/dsl/l0.py`) is an integer expression language. Recursive descent applies
multiplication before addition/subtraction and honors parentheses. There is no
division. Acceptance requires the entire expression to parse and execute.

L1 (`src/dsl/l1.py`) is a sequence of statements with no separator token. An
opcode marks the start of a statement. Expressions end at the next opcode.
The two core APIs are `set VAR expr` and `print expr`. Additional APIs are enabled
in this order: inc, dec, add, mul, clr, mov, neg, swap. Thus k counts **all** enabled
APIs, including set/print; k=6 means set, print, inc, dec, add, mul. Requested k is
clamped to 2 through 10 by the vocabulary helpers.

`inc x` and `dec x` update x by one; `add x y` and `mul x y` update x using y;
`clr x` sets x to zero; `mov x y` copies y to x; `neg x` negates x; `swap x y`
exchanges their values. All eight variables start at zero. Reading a variable
before an explicit write is legal. The interpreter returns success, final variable
values, printed outputs, and an error message on failure. It does not check a
program against a requested numerical answer.

The base interpreters tolerate BOS/EOS for compatibility. Evaluation version 2
validates the full token framing before passing the body to them; malformed
special tokens cannot be hidden by interpreter preprocessing.

## 3. Hardness and the withheld composition

`src/dsl/hard_l1.py` defines the hard sampler and shared gates. Freeze defaults:
k=6, six sampling variables, target statement range 5 through 9, expression-depth
configuration 3, and at least three counted cross-statement dependencies. Default
minimum distinct API count is zero (disabled). The withheld pair is add then mul.

`has_ordered_api_pair` checks whether any add opcode occurs before any mul opcode.
It does not require adjacency or that both operations share a variable. A reverse
mul-then-add sequence is allowed in training if no add precedes a later mul.
This is an opcode-order hold-out, not a data-flow-specific composition hold-out.

`count_cross_stmt_deps` counts statements that read a variable previously written.
It is a coarse proxy, not the length of a dependency chain. A statement contributes
at most one count even if it reads multiple previously written variables. Repeated
updates to one accumulator can accumulate many counts. The generic implementation
also treats clr/mov conservatively, so this counter should not be interpreted as
an exact semantic read-set analysis for future wider-k experiments.

Historical `--gate-knobs` overrides the target statement range to 10 through 16,
minimum counted dependencies to seven, and minimum distinct APIs to four. These
settings are explicitly recorded separately from the freeze constants.

Statement range and expression depth are generation targets, not strict grammar
limits. Templates and fallback padding may exceed the configured maximum statement
count, and templates use their own expression forms. Evaluation enforces the
minimum statement count, dependency count, and distinct-API requirement for hard
runs; it does not impose a maximum statement count or expression-depth ceiling.
All programs are bounded by the configured model context during evaluation.

## 4. Dataset construction and sampling arms

`src/data/generate.py` returns a DatasetSplit containing train, test_iid,
test_holdout, and metadata. Each program is stored as `[BOS, body..., EOS]`.
JSONL files contain one object per line with a `tokens` array.

Soft L1 uses procedural `generate_l1`. Train and IID reject programs containing
both members of a chosen pair; held-out programs must contain both, in either
order. For k=3 the default pair is print/inc. For k>=4 it is the third/fourth
enabled APIs. For k=2 it is set/print. This legacy split differs from hard L1's
ordered rule. L0 has only train and IID splits.

Hard L1 uses `src/dsl/templates_hard_l1.py` as a proposal frontend. Template families
are pipeline, fan-in, variable reuse, and holdout combination. Variables and digits
are randomized, and candidates may be padded toward the configured gates. The
shared acceptance rules then check execution, dependencies, API diversity, and
hold-out membership. Rejected candidates are counted by reason. Procedural
fallbacks propose alternatives; duplicate streaks trigger stronger diversity
fallbacks. This rejection process changes the realized distribution, so inspect
`sample_stats` rather than assuming uniform template frequencies.

The hard arm draws all training programs from the hard k=6 pool without ordered
add-before-mul. The mixture arm randomly selects k' from {2,3,6} for each training
proposal. Its smaller-k branches use easier procedural programs; its k=6 branch
uses hard sampling. This is a stationary random mixture, not a schedule that
becomes harder over epochs. Both arms evaluate on the hard k=6 test distributions.

L1 maintains a shared set of token tuples across training and both test splits,
so exact program duplicates cannot cross these splits. Mixture fallback programs
may change the effective training mix. Both IID filling paths have attempt limits
and fail explicitly if the requested number cannot be gathered. L0 uses a legacy
bounded uniqueness retry with a fallback that can repeat a program; do not claim
strict L0 train/test deduplication without changing that sampler.

Defaults are 200 examples per L1 test split. The N-curve driver chooses between 40
and 100 based on N. `--max-eval` scores only the first examples up to its cap. The
test sets are generated after training data using the same local Python RNG; they
therefore change when N or the sampling arm changes. Fixed external test datasets
are recommended for stronger comparisons across an N grid.

Saved dataset files are train.jsonl, test_iid.jsonl, test_holdout.jsonl, meta.json,
manifest.json, and vocab.json. Metadata records seed, arm, enabled APIs, the pair,
a hash of its ordered definition, freeze defaults, runtime knobs, and sampling
statistics. The manifest is a compact subset. The pair hash identifies the split
rule, not the actual dataset contents.

## 5. Model and training

`src/model/transformer.py` implements MiniGPT. Defaults are four pre-layer-norm
Transformer blocks, embedding width 128, four attention heads, dropout 0.1,
learned token and positional embeddings, and a final vocabulary projection.
Attention uses an explicit lower-triangular causal mask. Each feed-forward block
expands to four times the embedding width and uses GELU. Token embeddings and
output projection are separate parameters. The run records the exact parameter
count; this is safer than relying on approximate counts in comments.

`src/train.py` supplies TokenDataset and the CLI. A program is shifted into
input `seq[:-1]` and target `seq[1:]`. Inputs pad with PAD; target padding is -100
and ignored by cross entropy. BOS is an input, and EOS is a learned target.
Block size includes the complete stored sequence, including BOS/EOS. Oversized
training programs raise an actionable error rather than silently losing their
suffix. The CLI default block size is 128.

Training shuffles batches and applies AdamW at learning rate 3e-4, default batch
size 32, gradient clipping at norm 1.0, and a fixed number of epochs. AdamW's
unspecified settings use PyTorch defaults. There is no scheduler, early stopping,
validation-loss selection, resume support, or intermediate checkpointing. Logged
loss is the average of batch losses, not an exact token-weighted corpus loss.

Python and PyTorch are seeded at startup. Sampling during evaluation uses a local
Torch generator derived deterministically from run seed, epoch, and split; it
preserves the training RNG stream. Deterministic kernels are not forced, so
bitwise identity across hardware/PyTorch versions is not guaranteed.

Without `--data-dir`, omitting --arm preserves the toy soft path and honors
--api-k. `--arm hard` or `--hard` selects fixed k=6; `--arm mixture` selects the
mixture. `--hard --arm soft` is rejected. With --data-dir, dataset metadata controls
level and k, and the actual loaded training count is saved in checkpoint metadata.
The full loaded split is used; --n-train does not subsample an external dataset.

## 6. Evaluation version 2

`src/dsl/verification.py` contains the Torch-independent scoring contract.
`src/eval.py` runs the model and writes records.

Teacher forcing runs each reference program through the model once. At each
prefix, `src/dsl/continuations.py` returns structurally possible next tokens.
`legal_next_rate` is the fraction of scored positions whose argmax lies in that
set. `forced_unique_acc` restricts the denominator to positions with exactly one
legal continuation. Empty legal sets are skipped. These are structural metrics,
not reference-token accuracy or output correctness. Full reference inputs that
exceed model context raise an explicit error; they are never silently truncated.

For sampled L1 completion, `completion_prefix` supplies BOS and the reference's
first complete statement. If the reference has just one statement, it supplies
only BOS and the opcode, avoiding disclosure of the entire answer. L0 supplies
BOS because it has no statement boundary or hold-out split. Each test example
produces one temperature-1 sample; no grammar mask or rejection/resampling is
used. Reference suffixes are not copied or used to select generated tokens.

The default generation budget is the block size, capped per example by the space
remaining after its prefix. This replaces the old fixed 48-token cutoff and keeps
the entire generated program within the model's positional context. EOS ends
sampling. A sample that reaches its budget without EOS fails even if its current
body happens to parse. Direct `execution_accept_rate` callers can override the
budget or choose unconditional BOS generation; split evaluation always opts into
prefix completion.

A sample must have exactly one initial BOS, a terminal EOS, and no BOS/EOS/PAD/UNK
inside its body. `interpreter_accept_rate` counts correctly framed programs that
execute. `exec_accept_rate` additionally enforces task membership:

- Hard IID: no ordered withheld pair, plus minimum dependencies, distinct APIs,
  and statement count from dataset metadata.
- Hard hold-out: ordered withheld pair present, plus the same difficulty minima.
- Soft IID: the two withheld APIs must not both occur.
- Soft hold-out: both withheld APIs must occur.
- L0: correctly framed executable arithmetic expression.

The dataset's meta.json is required so scoring cannot silently drop its task
constraints. `n_unique_programs` counts distinct sampled bodies, including failed
samples; it is a diversity diagnostic, not proof of absence of mode collapse or
memorization. Many different programs can share the same opcode structure.

No version-1 result should be compared directly with version 2. Previously the
split execution loop ignored the reference program, generated twice from BOS,
stripped malformed special tokens, accepted missing EOS, and checked only the
base interpreter. Existing results files remain historical; rerun checkpoints
with the current evaluator or retrain to obtain version-2 metrics.

## 7. Saved run artifacts and replay

Training saves checkpoint.pt (weights, model config, vocabulary, metadata),
history.json (loss and periodic metrics), run_meta.json (arguments, actual training
count, artifact paths, evaluation version), a dataset subfolder, and a complete
train.jsonl copy even for external datasets. The checkpoint is written only after
all epochs and the final evaluation complete.

Legacy filenames free_gen_test_iid.jsonl and free_gen_test_holdout.jsonl now contain
**prefix completions** for L1. Latest files are overwritten at each evaluation.
Per-epoch `free_gen_<split>_epoch<E>.jsonl` files preserve individual evaluations.
History keeps five examples per split plus an examples_path to the complete file.
Each full record includes tokens, ok (task success), interpreter_ok, failure
reason, prefix_tokens, generated-token count, sampling_seed, evaluation_version,
and available run seed/epoch/split information. n_new_tokens excludes the prefix.

Standalone evaluation restores architecture and vocabulary from the checkpoint,
requires a matching dataset directory, and defaults to the checkpoint's seed and
training epoch for repeatable sampling. --seed and --epoch override those values.
Old checkpoints without epochs use epoch zero unless explicitly supplied. Standalone
--out writes completion dumps and a training-split copy, but prints the metrics;
it does not rewrite the original training history or run_meta.json.

## 8. Running experiments

Use Python >=3.11 with dependencies from requirements.txt. Run commands from the
repository root (the directory containing pyproject.toml):

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q

# Legacy soft pipeline, explicit for reproducibility
python -m src.train --level l1 --arm soft --api-k 3 --n-train 1000 \
  --epochs 15 --block-size 128 --out results/runs/soft_k3_n1000_v2

# Hard freeze configuration
python -m src.train --arm hard --n-train 1000 --epochs 15 \
  --block-size 128 --out results/runs/hard_n1000_v2

# Separate historical difficulty knobs, recorded in dataset metadata
python -m src.train --arm hard --gate-knobs --n-train 1000 --epochs 15 \
  --block-size 128 --out results/runs/hard_gate_n1000_v2

# Two-arm, fixed-k N curve
python -m src.experiments.n_curve --arms hard mixture \
  --n-grid 200 500 1000 --epochs 10 --block-size 128 \
  --out results/n_curve_k6_v2

# Re-evaluate a saved model using its dataset
python -m src.eval --checkpoint results/runs/hard_n1000_v2/checkpoint.pt \
  --data-dir results/runs/hard_n1000_v2/data --max-samples 200 \
  --out results/runs/hard_n1000_v2/reevaluation
```

`src/experiments/n_curve.py` trains a fresh model for each N and arm, then evaluates
at the final epoch. It uses one seed per invocation. Each cell saves result.json,
a checkpoint, its datasets, and completion dumps. Arm summaries and the curve
manifest report N*: the smallest tested N whose observed IID task success reaches
the threshold (default 0.9). None means no tested point crossed. There is no fitted
boundary, monotonicity constraint, multi-seed aggregation, or confidence interval.
With the default maximum 40 evaluated samples, uncertainty near 90% is substantial;
use larger tests and multiple seeds before making a threshold claim.

`job.sh` targets Bash/Linux/WSL with CUDA. diag checks visible GPUs and a CUDA
tensor; smoke runs one short cell; grid4 launches four cells on four GPUs for
N={1000,10000}, seeds {0,1}; grid4-seeds23 repeats seeds {2,3}. Each process sees
one GPU through CUDA_VISIBLE_DEVICES and trains for 15 epochs with gate knobs,
block size 128, and up to 80 samples per split. k in the directory name is fixed
at 6; it is not an implemented variable-k grid. A failed cell preserves its Python
exit code and causes the parallel grid to finish with a nonzero status. Logs are
written under logs/. Use new output directories for new runs to avoid mixing old
per-epoch files or overwritten checkpoints.

## 9. Verification and maintenance

Tests cover interpreter semantics, legal continuations, valid/disjoint L1 sampling,
hard hold-out rules, template sampling, 1k/10k gate generation, full-sample dumps,
framing failures, split-specific scoring, prefix selection, context rejection,
longer-than-48-token generation, RNG isolation, fresh L0 training, and GPU-cell
exit propagation. Torch-dependent tests require a working PyTorch installation.
The job-script tests use Bash with a mocked python3 command and need no GPU.

Validation on 2026-10-16: 40 non-Torch interpreter/data/scoring tests passed and
both mocked Bash cell tests passed. Python compilation and Bash syntax checking
passed. The Torch-dependent training/evaluation tests could not run locally:
PyTorch is absent and the managed network blocked dependency installation. Run
the complete suite and a short CUDA smoke job in the experiment environment
before launching an expensive grid. No GPU training results were regenerated.

When changing the experiment, update this guide, add a behavioral regression test,
and increment the evaluation version if scores change meaning. Preserve historical
result files, record actual runtime knobs and training count, and rerun affected
results instead of relabeling old values with a new scoring version.
