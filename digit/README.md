# unaveraged Digit

Reads Korean and English questions about phone-number digits and numeric street
addresses. The default model is **conv4-linear2 QAT**, with 4-bit convolution
and 2-bit grouped linear weights.

Use the main README for [building](../README.md#build),
[Chrome and Firefox installation](../README.md#install), and
[testing existing packages](../README.md#test), selecting `MODEL=digit`.
The [benchmark report](../benchmarks/README.md) contains model comparisons,
measurement conditions and the tradeoffs behind the default profile.

## Model and runtime

Images become grayscale float32 tensors of shape `[1, 1, 320, 672]`, normalized
to `[-1, 1]`. One forward pass produces 16 digit slots: 12 for a phone number and
four for a street number. The adapter selects the digits requested by the question.
A new question about a cached image can reuse the numeric record.

The default QAT recipe uses the bundled
[synthetic receipts and numeric annotations](../synth/README.md). The exact
training split, optimizer settings and backend options are documented in
[Build](../README.md#build). Fresh builds can produce different weights and
accuracy from the recorded benchmark artifacts.

The public FP32 source is
[tiny-receipt-reader-digit-slots-2m](https://huggingface.co/ivere27/tiny-receipt-reader-digit-slots-2m)
at revision `c9fa07886f99334ea820bfe312d6b4771d9460b0`.
Optional [artifact imports](../README.md#build) verify the exact files and
provenance against [`frozen-profiles.json`](frozen-profiles.json). Fresh builds
do not have to match those historical artifact hashes.

## Optional fresh W8A8 build

W8A8 converts the pinned public FP32 source and calibrates its 11 convolutions
with INT8 weights and activations; dense layers remain FP32. From the repository
root, use the bundled [calibration images](../synth/README.md):

```bash
make MODEL=digit PRECISION=w8a8 CALIBRATION_IMAGES=synth/images
```

For a separate calibration set, exclude external evaluation images explicitly:

```bash
make MODEL=digit PRECISION=w8a8 CALIBRATION_IMAGES=/path/to/calibration/images \
  EVAL_IMAGES=/path/to/evaluation/images
```

This writes `dist/digit/w8a8/`. Provide 256 distinct JPG/JPEG/PNG/WebP calibration
images directly inside the directory, or set `CALIBRATION_COUNT`. Public examples
and `EVAL_IMAGES` are excluded from calibration. The default seed is `20261027`.
Without `EVAL_IMAGES`, no external evaluation-overlap check is recorded.

Use the [test workflow](../README.md#test) with `MODEL=digit PRECISION=w8a8`.
Optional external evaluation is available through
`.venv/bin/python digit/check.py --evaluation-root /path/to/evaluation`, with
`images/` and `annotations/` subdirectories. External data and detailed reports
stay outside extension packages. Data handling is covered by the
[privacy policy](../PRIVACY.md).
