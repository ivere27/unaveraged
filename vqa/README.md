# unaveraged VQA

Answers Korean and English receipt questions about stores, phone numbers,
addresses, items, quantities and amounts. The default model is **W4A8 PTQ**,
with INT4 grouped linear weights and INT8 static convolutions.

Use the main README for [building](../README.md#build),
[Chrome and Firefox installation](../README.md#install), and
[testing existing packages](../README.md#test), selecting `MODEL=vqa`.
The [benchmark report](../benchmarks/README.md) contains model comparisons,
measurement conditions and the tradeoffs behind the default profile.

## Model and runtime

Images are grayscale 672×320. Questions use a 1,536-token byte-fallback BPE
vocabulary and are padded to 192 tokens with a mask. The encoder processes the
image and question, and the decoder generates one token at a time. Incomplete or
malformed outputs are not filled into the answer field. The result cache is keyed
by both image and question.

W4A8 uses group-32 INT4 weights and dynamic row-wise INT8 activations for its
67 dense layers. Static calibration affects the 13 front-end visual encoder
convolutions. The package includes a shared token embedding shard. This is a
post-training calibration recipe; it does not retrain the weights with QAT.
The default recipe uses the bundled [synthetic receipts](../synth/README.md),
as described in [Build](../README.md#build). Fresh calibration requires its own
accuracy checks; recorded phone/address results do not cover every supported
question family.

The public FP32 source is
[tiny-receipt-vqa-structured-qa-21m](https://huggingface.co/ivere27/tiny-receipt-vqa-structured-qa-21m)
at revision `f536c6ed181ca88929214f0bbc08f126f148c7ae`.
Optional [artifact imports](../README.md#build) require the exact graph, weights
and provenance in [`frozen-profiles.json`](frozen-profiles.json). Those historical
hashes are not requirements for a fresh build.

## Optional fresh PTQ build

From the repository root, use the bundled
[calibration images](../synth/README.md) and bilingual question templates:

```bash
make MODEL=vqa PRECISION=w8a8 CALIBRATION_IMAGES=synth/images
```

For separate calibration images and questions, exclude evaluation images:

```bash
make MODEL=vqa PRECISION=w8a8 CALIBRATION_IMAGES=/path/to/calibration/images \
  CALIBRATION_QUESTIONS=/path/to/questions.jsonl \
  EVAL_IMAGES=/path/to/evaluation/images
```

Provide 256 distinct JPG/JPEG/PNG/WebP images directly inside the calibration
directory, or set `CALIBRATION_COUNT`. Public examples and `EVAL_IMAGES` are
excluded. The default seed is `20261027`; without `EVAL_IMAGES`, no external
evaluation-overlap check is recorded. Questions are optional; bilingual templates
are used when omitted. JSONL records map a unique image basename to a question:

```jsonl
{"image":"receipt001.jpg","question":"What is the store phone number?"}
```

Fresh W8A8 calibration observes the encoder and FP32 decoder prefixes at positions
0, 4 and 12 without using target answers. Use `PRECISION=w4a8` to apply the same
input options to the default W4 profile. Each build replaces its precision's
cache/output. See [Build](../README.md#build) for cache reuse and the
[test workflow](../README.md#test) for checking a selected profile. These checks
do not establish accuracy on captured receipts. Data handling is covered by the
[privacy policy](../PRIVACY.md).
