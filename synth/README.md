# Initial synthetic receipt calibration set

256 synthetic PNG receipts for calibrating both the Digit and VQA models.
These are the exact **initial synthetic 256** images used in the 2026-10-27
comparison, copied without resizing, re-encoding or selecting different samples.
They were produced by the untrained initial rendering policy, before any RL
updates. Every image includes a watermark.

The images occupy 61,200,033 bytes (58.36 MiB). Paths in `manifest.json` are
relative to this directory; its record order is `000000` through `000255`.
Each record contains the original PNG hash, decoded RGB hash and dimensions,
plus the path and hash of its matching synthetic annotation JSON. The 256
annotation files occupy another 5,721,582 bytes (5.46 MiB). Post-training
calibration uses image pixels and model-generated decoder inputs; it does not
use the supplied answer labels.

Digit QAT uses numeric labels from the annotations; VQA PTQ uses image pixels
and generated decoder inputs. The default split and recipes are documented in
[Build](../README.md#build). See the [benchmark report](../benchmarks/README.md)
for recorded synthetic/real calibration comparisons and their evaluation limits.
This calibration set is not an independent accuracy benchmark and does not
qualify a fresh model build.

## Receipt contents and annotations

Each image has a matching JSON: `images/000000.png` pairs with
`annotations/000000.json`. These are unchanged generated labels from the original
synthetic receipts. Their two top-level sections are:

- `answers`: receipt contents, including `store`, `address`, `street_no`,
  `phone_disp`, `phone_digits` and `items`. Each item has `name`, `quantity`,
  `unit_price` and `total`. It also contains language and layout metadata.
  Some receipts have no store or item list; check `has_store` and `has_items`.
- `fields`: the rendered text fields and their polygons, including
  `glyphs[].char` and `glyphs[].poly` for individual characters.

For example, the content in `annotations/000000.json` includes:

```json
{
  "store": "싸요 클럽쉼터 직영점",
  "address": "오산시 예천군 옛군청옆길 868",
  "street_no": 868,
  "phone_disp": "161-1627",
  "phone_digits": "1611627",
  "has_items": false,
  "items": []
}
```

Coordinates refer to the original 670 × 320 PNG pixels. If you resize an image
to the models' 672 × 320 input, scale x coordinates by `672 / 670` when using
these polygons. Added watermark text is not part of `fields`. The renderer
changes appearance while preserving the receipt's geometry and label bytes.

## Use with this repository

From the repository root, first verify the downloaded files:

```bash
(cd synth && sha256sum --check --quiet SHA256SUMS)
```

Use the [main build instructions](../README.md#build) for default training,
calibration and cache reuse. Optional precision profiles and custom calibration
inputs are documented with each model:

- [Digit W8A8 calibration](../digit/README.md#optional-fresh-w8a8-build).
- [VQA PTQ calibration and question inputs](../vqa/README.md#optional-fresh-ptq-build).

For another quantizer, load the PNGs in manifest order and use the preprocessing
below. The VQA encoder also needs tokenized questions, attention masks and its
family-routing inputs; its decoder needs valid generated token/KV trajectories.
Images alone are not a complete VQA calibration feed. The original experiment's
two questions per image and decoder-prefix settings are recorded under
`reference_quantization` in the manifest.

## Preprocessing

Keep the distributed PNG files unchanged. At input time, both models use:

1. Convert to grayscale with Pillow `Image.convert("L")`.
2. Resize to width 672, height 320 with `Image.Resampling.BILINEAR`.
3. Convert to `float32` and normalize with `(pixel / 255.0 - 0.5) / 0.5`.
4. Arrange as NCHW; one image has shape `[1, 1, 320, 672]`.

The repository build applies this preprocessing automatically. The PNGs contain
their original RGB pixels and original dimensions, which can differ from the
model input size.

## Origin and scope

Paper/background pixels and font glyphs were rendered procedurally; captured
receipt images and crops were not composited into these images. Store names,
address prefixes and item names were sampled from a text bank built from existing
receipt annotations. Phone digits and street numbers were generated randomly.
The images therefore should not be described as using wholly invented text.
Watermarks were rendered with Noto Sans CJK Regular; font binaries are not bundled
in this dataset.

`manifest.json` records the generation stage, seed, source artifact hashes and
pinned FP32 model identities. The public files contain portable paths and include
only the matching generated synthetic annotations. Captured-receipt annotation
files, private evaluation data and experiment logs are excluded.

This is a calibration set, not an independent accuracy benchmark. The original
comparison used ONNX Runtime 1.23.2 static U8S8 QDQ; this repository's commands use
VolvoxAI with different conversion, question and decoder-prefix settings. Sharing
the same images does not imply identical quantized weights or the same accuracy.
The original comparison covered phone/address questions and does not establish
performance for all VQA task families.

## License

The distributed synthetic images, generated annotations and dataset
documentation/metadata are provided under the repository's
[Apache-2.0 license](../LICENSE).
See [NOTICE](../NOTICE) for project and third-party attribution.
