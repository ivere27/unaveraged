"""Import the pinned HF ONNX model and calibrate it with the public VolvoxAI API."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import onnx
import volvoxai as vx
from shared.model_support import (VERSION, sha256, preprocess,
    image_paths, packaged_library_paths, open_packaged_host, package_source,
    create_quantization_pipeline, export_quantization_plan, validate_quantized_weights,
    precision_metadata)

SOURCE_FORMAT = "receipt_digit_reader_onnx_v1"
PACKAGE_FORMAT = "volvoxai-receipt-digit-reader-onnx-package-v1"
INPUT_SHAPE = [1, 1, 320, 672]
OUTPUT_SHAPE = [1, 16, 11]
FLOAT_OPERATORS = ("BatchMatMul", "Linear", "GroupNorm", "SiLU", "LayerNorm", "Add")


def convert_model(source: Path, destination: Path, variant: str, version: str,
                  *, prepare_for_ptq: bool = False) -> None:
    config = json.loads((source / "config.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if config.get("format") != SOURCE_FORMAT or manifest.get("format") != SOURCE_FORMAT:
        raise ValueError("Unsupported Hugging Face receipt model format")
    if manifest.get("opset") != 18:
        raise ValueError("Expected the released opset-18 model")
    expected_decode = {"slots": 16, "phone_slots": 12, "street_slots": 4,
                       "num_classes": 11, "blank_class": 10}
    if any(config.get(key) != value for key, value in expected_decode.items()):
        raise ValueError("Receipt slot contract changed")
    if [config["input"].get(key) for key in ("channels", "height", "width")] != INPUT_SHAPE[1:]:
        raise ValueError("Receipt image contract changed")

    model_name = "model_int8.onnx" if variant == "int8" else "model.onnx"
    model = onnx.load(str(source / model_name), load_external_data=False)
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("Expected a self-contained ONNX release")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Expected one receipt image input and one slot output")

    # The published ONNX retains a symbolic slot axis even though the release
    # manifest fixes it at 16. Bind this and batch=1 through the public CLI.
    bounds = {}
    for value, name, shape, section in (
        (model.graph.input[0], "image", INPUT_SHAPE, "inputs"),
        (model.graph.output[0], "slot_logits", OUTPUT_SHAPE, "outputs"),
    ):
        expected = [{"name": name, "shape": ["batch", *shape[1:]], "dtype": "float32"}]
        if manifest.get(section) != expected:
            raise ValueError(f"Unexpected producer {section} contract")
        tensor = value.type.tensor_type
        if value.name != name or tensor.elem_type != onnx.TensorProto.FLOAT or len(tensor.shape.dim) != len(shape):
            raise ValueError(f"ONNX contract mismatch for {name}")
        for dimension, extent in zip(tensor.shape.dim, shape):
            if dimension.dim_param:
                symbol = dimension.dim_param
                if symbol in bounds and bounds[symbol] != extent:
                    raise ValueError(f"Inconsistent dimension {symbol}")
                bounds[symbol] = extent
            elif dimension.dim_value != extent:
                raise ValueError(f"Unexpected extent for {name}")

    destination.mkdir(parents=True)
    command = [sys.executable, "-I", "-m", "volvoxai", "export",
               "--model", str(source / model_name), "--out", str(destination / "model.safetensors"),
               "--input-shape", "image=1x1x320x672", "--output-name", "slot_logits",
               "--target", "wasm", "--quant-mode", "preserve",
               "--report", str(destination / "export-report.json"), "--report-format", "json"]
    for symbol, extent in sorted(bounds.items()):
        command += ["--dimension-bound", f"{symbol}={extent}:{extent}"]
    if prepare_for_ptq:
        # Use the published export option to expose SiLU as one float boundary
        # before PTQ, matching the receipt model's conservative PTQ recipe.
        command.append("--allow-silu-numerical-migration")
    # Isolated Python resolves the installed wheel, not a checkout on PYTHONPATH.
    subprocess.run(command, cwd=destination, check=True)

    graph = json.loads((destination / "graph.json").read_text())
    if graph.get("format") != "volvox-graph/v1" or graph.get("inputs") != {
        "input0": {"dtype": "float32", "shape": INPUT_SHAPE}
    } or graph.get("outputs") != ["slot_logits"]:
        raise ValueError("Converted model input/output contract changed")
    outputs = [output for node in graph["nodes"] for output in node.get("outputs", {}).values()
               if output.get("tensor") == "slot_logits"]
    if len(outputs) != 1 or outputs[0].get("shape") != OUTPUT_SHAPE or outputs[0].get("dtype") != "float32":
        raise ValueError("Converted slot output must be float32 [1, 16, 11]")

    package_manifest = {
        "format": PACKAGE_FORMAT, "variant": variant, "source_format": SOURCE_FORMAT,
        "source": {"sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                              for name in ("config.json", "manifest.json", model_name)}},
        "converter": {"package": "volvoxai", "version": version, "source": package_source(),
                      "prepare_for_ptq": prepare_for_ptq},
        "abi": {"input": {"name": "input0", "shape": INPUT_SHAPE}, "output": "slot_logits",
                "dimension_bounds": bounds,
                "batch": {"per_request": 1, "symbol": None, "min": 1, "max": 1, "multiple_of": 1}},
        "decode": expected_decode, "preprocess": config["input"],
    }
    (destination / "manifest.json").write_text(json.dumps(package_manifest, indent=2) + "\n")


def select_calibration(images: Path, excluded: list[Path], count: int, seed: int):
    if count < 1:
        raise ValueError("Calibration count must be positive")
    excluded_hashes = set()
    for directory in excluded:
        for path in image_paths(directory):
            excluded_hashes.add(sha256(preprocess(path).tobytes()))
    candidates = image_paths(images)
    random.Random(seed).shuffle(candidates)
    selected, seen = [], set()
    for path in candidates:
        normalized = sha256(preprocess(path).tobytes())
        if normalized in excluded_hashes or normalized in seen:
            continue
        seen.add(normalized)
        selected.append({"path": path, "file_sha256": sha256(path.read_bytes()),
                         "input_sha256": normalized})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Need {count} distinct calibration images outside evaluation; found {len(selected)}")
    return selected, excluded_hashes


def quantize_model(fp32: Path, destination: Path, *, images: Path,
                   excluded: list[Path], count: int, seed: int,
                   evaluation_checked: bool = False, source=None, questions=None,
                   precision='w8a8', cpu_threads=None) -> dict:
    settings = precision_metadata(precision)
    # A running build may have imported an older shared helper before this
    # module was loaded. Omitted options must retain its original call contract.
    thread_options = {} if cpu_threads is None else {'cpu_threads': cpu_threads}
    print("Select distinct calibration images and exclude evaluation inputs", flush=True)
    selected, excluded_hashes = select_calibration(images, excluded, count, seed)
    destination.mkdir(parents=True)
    engine, loader = packaged_library_paths()
    host = open_packaged_host()
    pb = vx.pb
    started = time.perf_counter()
    plan = None
    samples = 0
    input_digest = hashlib.sha256()
    try:
        quantization = vx.VxQuantizationServiceClient(host)
        plan, expected_nodes, stages = create_quantization_pipeline(
            quantization, fp32, precision, ('Conv2D',), **thread_options)
        if precision == 'w8a8' and expected_nodes != 11:
            raise ValueError(f'Expected 11 receipt convolutions; got {expected_nodes}')
        for index, item in enumerate(selected if plan.info.requires_calibration else []):
            values = preprocess(item["path"])
            raw = values.tobytes()
            if sha256(raw) != item["input_sha256"]:
                raise ValueError(f"Calibration image changed: {item['path']}")
            input_digest.update(raw)
            quantization.calibrate_quantization_plan(pb.CalibrateQuantizationPlanRequest(
                quantization_plan_id=plan.quantization_plan_id, profile_name="default", sample_name=f"receipt-{index}",
                sample_count=1, inputs=[pb.Tensor(name="input0", shape=INPUT_SHAPE,
                    dtype=pb.DataType.DATA_TYPE_F32, inline=raw)]))
            samples += 1
            if (index + 1) % 32 == 0 or index + 1 == count:
                print(f"VolvoxAI PTQ calibration {index + 1}/{count}", flush=True)
        graph = export_quantization_plan(quantization, plan, destination, samples)
    finally:
        try:
            if plan is not None:
                quantization.release_quantization_plan(pb.QuantizationPlanRef(
                    quantization_plan_id=plan.quantization_plan_id))
        finally:
            host.close()

    validate_quantized_weights(destination, graph, precision, expected_nodes)
    convolutions = [node for node in graph["nodes"] if node["opType"] == "QConv2D"]
    if len(convolutions) != 11 or any(node["opType"] == "Conv2D" for node in graph["nodes"]):
        raise ValueError("PTQ must retain no FP32 receipt convolutions")
    if not any(node["params"].get("stride") == [2, 1] for node in convolutions):
        raise ValueError("Missing the quantized asymmetric-stride convolution")
    if graph["inputs"] != {"input0": {"shape": INPUT_SHAPE, "dtype": "float32"}} or graph["outputs"] != ["slot_logits"]:
        raise ValueError("PTQ changed the public model ABI")
    output = [value for node in graph["nodes"] for value in node["outputs"].values()
              if value["tensor"] == "slot_logits"]
    if len(output) != 1 or output[0]["shape"] != OUTPUT_SHAPE or output[0]["dtype"] != "float32":
        raise ValueError("PTQ changed the slot output contract")

    identities = {
        "seed": seed, "count": count, "images_directory": str(images),
        "excluded_directories": [str(directory) for directory in excluded],
        "excluded_input_sha256": sorted(excluded_hashes),
        "images": [{**item, "path": str(item["path"])} for item in selected],
        "input_sha256": input_digest.hexdigest(),
    }
    (destination / "calibration-inputs.json").write_text(json.dumps(identities, indent=2) + "\n")
    metadata = {
        "method": "volvoxai-ptq", "source_variant": "fp32", "source_model": "model.onnx",
        "package": "volvoxai", "version": VERSION, "backend": "cpu",
        **settings, "quantized_nodes": expected_nodes, "stages": stages,
        "float_operators": sorted({node['opType'] for node in graph['nodes']
            if node['opType'] in FLOAT_OPERATORS or node['opType'] == 'Conv2D'}),
        "quantized_convolutions": len(convolutions),
        "packed_linear_nodes": sum(node['opType'] == 'PackedLinear' for node in graph['nodes']),
        "retained_float_convolutions": sum(node['opType'] == 'Conv2D' for node in graph['nodes']),
        "calibration": {"samples": samples, "required": plan.info.requires_calibration,
                        **thread_options,
                        "selected_images": count, "seed": seed, "coverage_complete": True,
                        "input_sha256": input_digest.hexdigest(),
                        "evaluation_checked": evaluation_checked,
                        "evaluation_overlap": 0 if evaluation_checked else None},
        "native_sha256": {engine.name: sha256(engine.read_bytes()), loader.name: sha256(loader.read_bytes())},
    }
    manifest = json.loads((fp32 / "manifest.json").read_text())
    manifest.update(variant="ptq", quantization=metadata)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {**metadata, "seconds": time.perf_counter() - started,
              "operators": dict(Counter(node["opType"] for node in graph["nodes"])),
              "source_package_sha256": {name: sha256((fp32 / name).read_bytes())
                                        for name in ("graph.json", "model.safetensors")}}
    (destination / "ptq-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return metadata
