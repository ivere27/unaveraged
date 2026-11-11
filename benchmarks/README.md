# Benchmarks

The defaults remain **Digit conv4-linear2 QAT** and **VQA W4A8 PTQ**.
[results.json](results.json) records the selected models; [digit-comparison.json](digit-comparison.json)
contains the current 13-model Digit comparison and artifact identities. Public
results contain aggregates only, without receipts or individual predictions.

The [published benchmark page](https://ivere27.github.io/unaveraged/benchmarks.html)
renders tables directly from these JSON files; `make docs` renders them locally.

All measurements use **VolvoxAI 0.9.0**. Default builds use its published
Python wheel (`volvoxai==0.9.0`) and `volvoxai-lite@0.9.0` npm package.

## Quick benchmark summary

### Inference throughput & latency

| Task / Model | Precision | Platform / Backend | Latency | 2,000 heldout accuracy |
| --- | --- | --- | --- | --- |
| **Digit** (`conv4-linear2`) | 4-bit Conv / 2-bit Linear | Native CPU (1 thread) | 28.8 ms / image | 98.40% (Ans) / 95.65% (Rec) |
| **Digit** (`conv4-linear2`) | 4-bit Conv / 2-bit Linear | Native CPU (Multi-thread) | 27.1 ms / image | 98.40% (Ans) / 95.65% (Rec) |
| **Digit** (`conv4-linear2`) | 4-bit Conv / 2-bit Linear | Chrome WASM (Extension) | 88.2 ms (median) | 98.40% (Ans) / 95.65% (Rec) |
| **Digit** (`w8a8`) | 8-bit Conv / 8-bit Linear | Native CPU (1 thread) | 32.3 ms / image | 98.65% (Ans) / 96.80% (Rec) |
| **VQA** (`w4a8`) | 8-bit Conv / 4-bit Linear | Native CPU (Multi-thread) | 168.5 ms / query | 97.15% (1,943/2,000) |
| **VQA** (`w4a8`) | 8-bit Conv / 4-bit Linear | Chrome WASM (Extension) | 1,514.2 ms / query | 97.15% (1,943/2,000) |
| **VQA** (`w8a8`) | 8-bit Conv / 8-bit Linear | Native CPU (Multi-thread) | 214.5 ms / query | 97.25% (1,945/2,000) |
| **VQA** (`w8a8`) | 8-bit Conv / 8-bit Linear | Chrome WASM (Extension) | 1,689.9 ms / query | 97.25% (1,945/2,000) |

*Notes*: Digit evaluations test 2,000 heldout receipt images. VQA evaluations test 2,000 heldout questions across store, address, and phone query types. Chrome WASM latencies are measured under headless Chromium on production extension builds.

### Training & quantization throughput

| Pipeline | Model / Precision | Hardware / Backend | Step time | Total job time |
| --- | --- | --- | --- | --- |
| **Digit QAT** (2 epochs, 256 steps) | `conv4-linear2` Joint Weight QAT | CPU (1 thread) | 21.0 s / step | ~90 min |
| **Digit QAT** (2 epochs, 256 steps) | `conv4-linear2` Joint Weight QAT | CPU (2 threads) | 12.5 s / step | ~53 min |
| **Digit QAT** (2 epochs, 256 steps) | `conv4-linear2` Joint Weight QAT | CPU (10 threads) | 14.7 s / step | ~63 min |
| **Digit QAT** (2 epochs, 256 steps) | `conv4-linear2` Joint Weight QAT | GPU (CUDA, RTX 3090) | **0.74 s / step** | **~3.2 min (28× speedup)** |
| **Digit PTQ** (256 calibration images) | `w8a8` Static INT8 PTQ | CPU (Multi-thread) | — | ~10 sec |
| **VQA PTQ** (256 calibration images) | `w4a8` Dynamic INT8 / INT4 | CPU (Multi-thread) | — | ~1.5 - 2.0 min |

*Notes*: Digit QAT trains on 224 bundled synthetic receipts with 32 reserved for validation (batch size 4). GPU training leverages PyPI `volvoxai` with CUDA acceleration.


## Digit comparison

Every model was evaluated on the same **2,000 heldout records**
(1,986 distinct image contents). Complete-record accuracy requires
both the phone number and street number to match. Question-answer accuracy
compares only the value requested by that record's question, so it can be
higher even when other digits are wrong. Both accuracy measures match in native CPU and Chrome.
Accuracy was measured on a Ryzen 5 5600U; the latency columns use the separate
Core i3-1115G4 interleaved run, with 45 executions per model.

<!-- benchmark:digit-comparison -->

Conv/Linear numbers describe weight bits. Quantized convolution activations are
static INT8; packed dense activations are dynamic INT8 with group size 32.
W8A8 retains FP32 dense layers. Other operations retain floating point.
Importance and MSE PTQ are distinct calibration recipes. Joint QAT trained
convolution and dense weights; head-only QAT kept convolutions fixed. QAT used
weight fake quantization, followed by activation calibration, without activation
QAT. The same selected Conv4/Linear2 QAT artifact appears only once.

The selected **epoch-2 Conv4/Linear2 QAT** artifact prioritizes package size.
The comparison includes its same-size MSE PTQ counterpart and the larger W8A8
reference, so accuracy and latency tradeoffs remain visible. MiB = 1,048,576
bytes; Digit size covers graph and weights, excluding runtime and ZIP compression.
The recorded mixed-data QAT artifact differs from the default public-only fresh
training recipe; building that profile does not reproduce its heldout score.

## Digit timing and validation

Accuracy uses every heldout record on an **AMD Ryzen 5 5600U**.
Primary latency runs separately on an **Intel Core i3-1115G4** (2 cores,
4 hardware threads), with batch 1 and one CPU/WASM numerical thread.
The latency browser is Chrome/150.0.7871.24.
It uses an interleaved run of the same three heldout inputs
(indices 0, 999 and 1,999), with 15 cycles and 45 measured executions per model.
Each model/input receives one reference execution and four warmups before timing.
Model execution order alternates between cycles, keeping the inputs fixed and
balancing execution order across models. Native and Chrome timing runs were
serial on the i3 host. Full-heldout accuracy-pass timings are auxiliary data in
JSON; the tables use the interleaved measurements.

<!-- benchmark:digit-timing -->

Timers include execution and input/output transport, excluding file reads,
preprocessing, hashing, scoring, loading/compilation, warmups and UI/queue work.
Exact timer boundaries and warmup conditions are retained in the JSON protocol.
Chrome additionally compares all outputs with the corresponding native logits
and slots. Per-model parity, tolerances, runtime/model hashes and phone/street
accuracy counts are retained in `digit-comparison.json`. P95 uses linear
interpolation across the 45 executions; this is not a full-heldout latency
distribution or production latency guarantee. The host is not certified idle.
This run measures CPU/WASM, not CUDA inference.

## Digit native/Chrome agreement

On the accuracy host, all 16 output slots match between native and Chrome for every model/input pair. Some logits exceed the recorded numerical tolerance (rtol 0.0005, atol 0.001); matching accuracy or slots does not imply numerical parity.

<!-- benchmark:digit-parity -->

On the separate latency host, all three inputs match the saved accuracy-host native slots and stay within the recorded logit tolerance.

## VQA timing

Measured on an **AMD Ryzen 5 5600U** with VolvoxAI 0.9.0, using three public
questions on one receipt. Full-heldout VQA accuracy is not measured.

<!-- benchmark:vqa-size -->

The selected W4 package uses W4/group32 dense weights with dynamic INT8
activations; convolutions retain static INT8. W8 has a different activation and
calibration recipe, so this comparison does not isolate weight bit width.

<!-- benchmark:vqa-timing -->

Three public questions on one English receipt, one warmup and five cycles give
15 observations per model. Answer timing includes tokenization, encoder,
autoregressive decoding, transfers and output cleanup, excluding preprocessing,
loading and UI work. Fixed-prefix timing holds 16 tokens/cache lengths constant;
actual generation length varies. Independently pooled stage medians need not
sum to the complete-answer median. MB = 1,000,000 bytes; VQA package size includes
graphs, weights, shared embedding once, vocabulary, configuration and manifest,
excluding runtime and ZIP.

## Scope and reproduction

The heldout corpus has prior development/evaluation use and the supplied labels
are not a fully human-audited gold set. A fresh runtime evaluation does not make
it an independent model-selection test. Digit reports both whole-record and
requested-answer accuracy.

Actual page response also adds preprocessing, a 100 ms debounce, messages and
queueing. Digit caches by image; VQA caches by image and question. Cache hits skip
inference.

The full-corpus accuracy commands use the private fixtures and official runtime
packages prepared on the Ryzen 5 5600U. Models, runtimes, reports and native
references are retained. The temporary 2,000-input pixel cache is removed after
verification to recover disk space; the original heldout and training data are
unchanged. Before rerunning, regenerate those tensors from the original heldout
images with the same preprocessing and verify every input against the saved
fixture hash, preserving the existing fixture and audit records. Private inputs
are excluded from the source bundle.
The retained local `restore_inputs.py` helper restores only missing tensors,
checks all 2,000 source/input hashes and refuses changed existing files. It
preserves fixture and audit metadata; `--verify-only` checks without writing.
The helper and its fixtures remain private and are not in the source bundle.

```sh
.venv/bin/python .cache/benchmarks/digit-upstream/restore_inputs.py

PYTHONPATH="$PWD/.cache/benchmarks/digit-upstream/python-runtime" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python benchmarks/digit_native_eval.py \
  --fixtures .cache/benchmarks/digit-upstream/fixtures.private.json \
  --runtime-version 0.9.0 --out .cache/benchmarks/digit-upstream/native --resume

node benchmarks/digit_wasm_eval.mjs \
  --fixtures .cache/benchmarks/digit-upstream/fixtures.private.json \
  --runtime-dir .cache/benchmarks/digit-upstream/npm/package/dist/0.9.0 \
  --runtime-version 0.9.0 --runtime-source published-npm \
  --out .cache/benchmarks/digit-upstream/wasm --resume
```

The full-corpus runners' `--resume` validates saved identities before reusing
completed results. Use new output directories for a new measurement and point
the fixture's native reference paths to that run.

Primary latency ran serially on the Core i3-1115G4 from an isolated staging
directory. It contains the benchmark scripts, `latency-fixtures/` with the three
private inputs, frozen models and native references; the verified wheel in
`python-runtime/`; NumPy/protobuf in `python-deps/`; the official npm JS/WASM in
`runtime/`; and the recorded Node/Chrome installations and Node dependencies.
From that staging root:

```sh
PYTHONPATH="$PWD/python-runtime:$PWD/python-deps" PYTHONNOUSERSITE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -S benchmarks/digit_latency.py \
  --fixtures latency-fixtures/fixtures.private.json --runtime-version 0.9.0 \
  --out latency/interleaved-native.json

RECEIPT_CHROME="$PWD/chrome/chrome" bin/node benchmarks/digit_latency.mjs \
  --fixtures latency-fixtures/fixtures.private.json --runtime-dir runtime \
  --runtime-version 0.9.0 --runtime-source published-npm \
  --out latency/interleaved-chrome.json
```

Latency runners require a new output filename and verify fixture, model and
runtime hashes. The three latency inputs are retained; their private subset
fixture is excluded from public outputs.
See the [main README](../README.md) for extension build and functional checks.

### VQA public timing

Use the recorded FP32, W8A8 and W4A8 packages to reproduce the measured model
identities. Freshly converted packages are a new experiment. The fixture builder
requires explicit package paths, records their hashes and saves native references
for the same three public questions and fixed decoder prefix.

```sh
.venv/bin/python benchmarks/vqa_fixtures.py \
  --fp32-package /path/to/recorded/fp32 \
  --w8a8-package /path/to/recorded/w8a8 \
  --w4a8-package /path/to/recorded/w4a8 \
  --out .cache/benchmarks/vqa-fixtures

node benchmarks/vqa_wasm.mjs \
  --fixtures .cache/benchmarks/vqa-fixtures/fixtures.json \
  --out .cache/benchmarks/vqa-timing \
  --models w4a8,w8a8 --cycles 5 --warmups 1 --fixed-steps 16
```

Use new output directories for each measurement. Full-heldout VQA accuracy is
outside this public timing run. Historical batch, CUDA, repacking and low-bit
search experiments are no longer included; current model construction lives in
`digit/` and `vqa/`.
