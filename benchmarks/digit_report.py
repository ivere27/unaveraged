#!/usr/bin/env python3
"""Consolidate completed Digit heldout runs without publishing private records."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import struct
import tarfile
from urllib.parse import urlparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_COUNT = 13
SAMPLES = 2000
SELECTED = 'conv4-linear2'
FINGERPRINTS = ('ordered_inputs_sha256', 'ordered_labels_sha256',
                'ordered_image_files_sha256', 'ordered_annotation_files_sha256')
SCORE_KEYS = ('record_correct', 'phone_correct', 'street_correct', 'answer_correct')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def digest_order(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def decode_slots(slots):
    require(len(slots) == 16 and all(type(value) is int and 0 <= value <= 10 for value in slots), 'Invalid saved slots')
    def digits(part):
        return ''.join(str(value) for value in part[:part.index(10) if 10 in part else len(part)])
    return {'phone': digits(slots[:12]), 'street': digits(slots[12:])}


def reference_record(reference):
    values = reference['values']
    require(len(values) == 176 and all(math.isfinite(value) for value in values), 'Native reference is not 176 finite logits')
    require(sha(struct.pack('<176f', *values)) == reference['sha256'], 'Native reference bytes/hash differ')
    slots = [max(range(11), key=lambda column: values[row * 11 + column]) for row in range(16)]
    require(slots == reference['slots'], 'Native reference logits/slots differ')
    return decode_slots(slots)


def latency_stats(values, expected_samples=SAMPLES):
    require(len(values) == expected_samples and all(math.isfinite(value) and value > 0 for value in values), 'Incomplete/invalid latency samples')
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower, upper = math.floor(position), math.ceil(position)
    return {'samples': len(values), 'median_ms': statistics.median(values), 'mean_ms': statistics.mean(values),
            'p95_ms': ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)}


def same_stats(actual, expected, label):
    require(actual['samples'] == expected['samples'], label + ': latency sample count differs')
    require(all(math.isclose(actual[key], expected[key], rel_tol=1e-10, abs_tol=1e-8)
                for key in ('median_ms', 'mean_ms', 'p95_ms')), label + ': latency statistics differ from private samples')


def accuracy(counts):
    require(set(counts) == set(SCORE_KEYS) and all(type(value) is int and 0 <= value <= SAMPLES for value in counts.values()),
            'Invalid accuracy counts')
    return {**counts, 'samples': SAMPLES, 'record_percent': 100 * counts['record_correct'] / SAMPLES,
            'answer_percent': 100 * counts['answer_correct'] / SAMPLES}


def safe_evidence_path(directory, value):
    target = (directory / value).resolve()
    require(target.parent == directory.resolve() and target.is_file(), 'Unexpected evidence file path')
    return target


def verify_upstream(run, upstream):
    version = upstream['version']
    require(version == '0.9.0', 'This heldout comparison requires upstream version 0.9.0')
    npm, python = upstream['npm'], upstream['python']
    require(npm['name'] == 'volvoxai-lite' and python['name'] == 'volvoxai'
            and npm['version'] == python['version'] == version, 'Upstream package identities differ')
    require(npm['registry'] == 'https://registry.npmjs.org' and python['source'] == 'pypi', 'Unexpected package source')
    require(urlparse(npm['dist']['tarball']).hostname == 'registry.npmjs.org'
            and urlparse(python['url']).hostname == 'files.pythonhosted.org', 'Expected official package download URLs')
    tarball = run/'downloads'/Path(urlparse(npm['dist']['tarball']).path).name
    wheel = run/'downloads'/python['wheel']
    tarbytes, wheelbytes = tarball.read_bytes(), wheel.read_bytes()
    require(sha(tarbytes) == npm['tarball_sha256'] and hashlib.sha1(tarbytes).hexdigest() == npm['dist']['shasum'],
            'Downloaded npm archive differs from registry identity')
    integrity = 'sha512-' + base64.b64encode(hashlib.sha512(tarbytes).digest()).decode()
    require(integrity == npm['dist']['integrity'], 'Downloaded npm archive integrity differs')
    require(sha(wheelbytes) == python['sha256'], 'Downloaded wheel differs from PyPI identity')
    with tarfile.open(tarball, 'r:gz') as archive:
        package = json.load(archive.extractfile('package/package.json'))
        require(package['name'] == npm['name'] and package['version'] == version, 'npm package metadata differs')
        for name, expected in npm['sha256'].items():
            require(sha(archive.extractfile(f'package/dist/{version}/{name}').read()) == expected, 'npm runtime differs from tarball')
            require(sha((run/'npm/package/dist'/version/name).read_bytes()) == expected, 'Installed npm runtime changed')
    native_hashes = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in ('libvolvoxai.so', 'libsynurang_module_host.so', '__init__.py', '_library.py', '_session.py'):
            native_hashes[name] = sha(archive.read('volvoxai/' + name))
            require(sha((run/'python-runtime/volvoxai'/name).read_bytes()) == native_hashes[name], 'Installed native runtime differs from wheel')
    return {'package': 'volvoxai', 'version': version, 'source': 'pypi', 'sha256': native_hashes,
            'wheel': python['wheel'], 'wheel_sha256': python['sha256'], 'url': python['url']}, {
            'package': 'volvoxai-lite', 'version': version, 'source': 'published-npm', 'sha256': npm['sha256'],
            'tarball_sha256': npm['tarball_sha256'], 'integrity': npm['dist']['integrity'], 'url': npm['dist']['tarball']}


def public_split_audit(run, fixture, models):
    audit_path = run/'split-audit.json'
    audit = read(audit_path)
    require(audit['format'] == 'unaveraged-digit-split-audit/v1'
            and audit['status'] == 'passed-with-partial-coverage', 'Unexpected split audit status')
    require(audit['heldout']['records'] == SAMPLES
            and audit['heldout']['ordered_inputs_sha256'] == fixture['ordered_inputs_sha256']
            and audit['heldout']['ordered_labels_sha256'] == fixture['ordered_labels_sha256'], 'Split audit corpus differs')
    require(audit['new_training_performed'] is False and audit['reused_frozen_models'] == MODEL_COUNT
            and set(audit['coverage']) == set(models), 'Split audit model coverage differs')
    stages = {}
    for name, entry in audit['audits'].items():
        require(entry['status'] == 'audited-zero-overlap'
                and entry['overlap_unique_file_hashes'] == entry['overlap_unique_input_hashes'] == 0
                and entry['file_hash_coverage_complete'] and entry['input_hash_coverage_complete'],
                'An audited stage no longer has complete zero-overlap hash evidence')
        stages[name] = {'samples': entry['rows'], 'manifest_sha256': entry['manifest_sha256'],
                       'overlap_unique_file_hashes': 0, 'overlap_unique_input_hashes': 0}
    coverage = {}
    for name, entry in audit['coverage'].items():
        require(entry['base_fp32_pretraining']['status'] == 'unknown'
                and entry['new_training_or_calibration'] is False, 'Original pretraining must remain explicitly unaudited')
        linked = {}
        for stage, detail in entry['stages'].items():
            linked[stage] = {'status': detail['status']}
            if 'audit' in detail:
                require(detail['audit'] in stages, 'Missing linked split audit')
                linked[stage]['audit'] = detail['audit']
        coverage[name] = {'stages': linked, 'coverage': 'listed-stages-only' if linked else 'not-linked-in-this-audit'}
    return {'status': audit['status'], 'scope': 'Stored historical file/pixel hash manifests compared with the current heldout; source image bytes were not reread.',
            'new_training_performed': False, 'base_fp32_pretraining': 'unknown',
            'audited_zero_overlap_stages': stages, 'model_coverage': coverage,
            'limitations': [
                'Audit coverage is partial. Empty model stage lists and historical-aggregate-only entries are not newly verified zero overlap.',
                'Original FP32 pretraining is unaudited; no complete training-lifecycle leakage guarantee is claimed.',
                'These frozen models were previously examined on the same corpus, so this is not an independent blind evaluation.',
                'Historical QAT used real and synthetic data. No new training or calibration used the currently supplied training path.',
                'Zero exact-file/pixel hash overlap does not establish semantic-scene or near-duplicate independence.']}, sha(audit_path.read_bytes())


def interleaved_latency(run, fixture, fixture_sha, models, native_runtime, wasm_runtime):
    subset_path = run/'latency-fixtures/fixtures.private.json'
    subset = read(subset_path)
    indices = [0, 999, 1999]
    input_hashes = [fixture['cases'][index]['sha256'] for index in indices]
    require(subset['format'] == 'unaveraged-digit-latency-fixtures/v1'
            and subset['source_fixture_sha256'] == fixture_sha and subset['source_indices'] == indices,
            'Latency subset lineage differs')
    require([item['source_index'] for item in subset['cases']] == indices
            and [item['sha256'] for item in subset['cases']] == input_hashes
            and subset['cycles'] == 15 and subset['warmups'] == 4
            and subset['atol'] == fixture['atol'] and subset['rtol'] == fixture['rtol'], 'Latency subset/protocol differs')
    require(set(subset['models']) == set(models), 'Latency subset lacks a comparison model')
    stats, protocols, identities, warnings, checks, environments, hosts = {}, {}, {}, [], {}, {}, {}
    for engine, filename, expected_runtime in [('native', 'interleaved-native.json', native_runtime),
                                              ('chrome', 'interleaved-chrome.json', wasm_runtime)]:
        report_path = run/'latency'/filename
        report = read(report_path)
        require(report['format'] == 'unaveraged-digit-interleaved-latency/v1'
                and report['engine'] == ('native-cpu' if engine == 'native' else 'chrome')
                and report['status'] in ('passed', 'completed-with-numerical-differences'), 'Incomplete interleaved timing')
        require(report['runtime']['version'] == expected_runtime['version']
                and report['runtime']['sha256'] == expected_runtime['sha256'], 'Interleaved timing runtime differs')
        require(report['fixture_sha256'] == sha(subset_path.read_bytes()) and report['input_sha256'] == input_hashes
                and report['input_indices'] == indices and report['source_fixture_sha256'] == fixture_sha
                and report['source_references_sha256'] == subset['source_references_sha256'],
                'Interleaved timing inputs differ')
        require(set(report['models']) == set(models), 'Interleaved timing lacks a comparison model')
        host = report['environment']
        require(isinstance(host.get('cpu'), str) and host['cpu'] != 'unknown'
                and isinstance(host.get('hostname'), str) and host['hostname']
                and type(host.get('logical_cpus')) is int and host['logical_cpus'] > 0,
                'Interleaved timing host identity is missing')
        hosts[engine] = (host['hostname'], host['cpu'], host['logical_cpus'])
        environments[engine] = {key: host[key] for key in
            ('cpu', 'logical_cpus', 'platform', 'python', 'numpy', 'node') if key in host}
        observations = report['host_observations']
        require([row['phase'] for row in observations] == ['before', 'after']
                and all(len(row['loadavg']) == 3 and all(math.isfinite(value) and value >= 0 for value in row['loadavg'])
                        for row in observations), 'Interleaved host-load observations are incomplete')
        environments[engine]['host_observations'] = observations
        protocol = report['protocol']
        require(protocol['cycles'] == 15 and protocol['warmups_per_input'] == 4 and protocol['input_count'] == 3
                and protocol['samples_per_model'] == 45 and protocol['batch_size'] == protocol['cpu_threads'] == 1,
                'Unexpected interleaved timing protocol')
        require(isinstance(protocol['order'], str) and 'revers' in protocol['order'].lower(),
                'Interleaved model order must reverse each cycle')
        required_sources = ({'benchmarks/digit_latency.py', 'benchmarks/digit_native_eval.py'} if engine == 'native'
                            else {'benchmarks/digit_latency.mjs', 'benchmarks/runtime-benchmark.js', 'benchmarks/worker.js'})
        require(set(report['sources_sha256']) == required_sources, 'Interleaved timing caller identity missing')
        for source, expected in report['sources_sha256'].items():
            require(not Path(source).is_absolute() and '..' not in Path(source).parts
                    and sha((ROOT/source).read_bytes()) == expected, 'Interleaved timing source differs')
        engine_stats, engine_checks = {}, {}
        for name, model in models.items():
            measured, prepared = report['models'][name], subset['models'][name]
            require(measured['sha256'] == prepared['sha256'] == model['sha256']
                    and measured['packed_arithmetic'] == prepared['packed_arithmetic'] == model['packed_arithmetic'],
                    'Interleaved timing model differs')
            values = measured['samples_ms_per_input']
            require(len(values) == 3 and all(len(row) == 15 for row in values), 'Interleaved timing sample counts differ')
            summary = latency_stats([value for row in values for value in row], 45)
            require(math.isclose(measured['pooled_median_ms'], summary['median_ms'], rel_tol=1e-10, abs_tol=1e-8)
                    and math.isclose(measured['mean_ms'], summary['mean_ms'], rel_tol=1e-10, abs_tol=1e-8)
                    and math.isclose(measured['p95_ms'], summary['p95_ms'], rel_tol=1e-10, abs_tol=1e-8),
                    'Interleaved timing aggregates differ from samples')
            engine_stats[name] = summary
            if engine == 'native':
                comparisons = measured['parity']
                require(len(comparisons) == 3 and sorted(row['input_index'] for row in comparisons) == [0, 1, 2],
                        'Interleaved native parity is incomplete')
                for comparison in comparisons:
                    reference = subset['cases'][comparison['input_index']]['references'][name]
                    require(comparison['reference_sha256'] == reference['sha256']
                            and comparison['bytes_equal'] == (comparison['output_sha256'] == reference['sha256'])
                            and math.isfinite(comparison['maximum_absolute_logit_difference'])
                            and comparison['maximum_absolute_logit_difference'] >= 0, 'Interleaved native reference evidence differs')
                native_check = {'native_reference_bytes_equal': all(row['bytes_equal'] for row in comparisons),
                    'native_reference_slots_equal': all(row['slots_equal'] for row in comparisons),
                    'native_reference_logits_within_tolerance': all(row['logits_within_tolerance'] for row in comparisons),
                    'maximum_absolute_logit_difference': max(row['maximum_absolute_logit_difference'] for row in comparisons)}
                require(all(measured[key] == value for key, value in native_check.items()), 'Interleaved native parity summary differs')
                engine_checks[name] = native_check
            else:
                comparisons = [row for row in report['parity'] if row['profile'] == name]
                require(len(comparisons) == 3 and sorted(row['input_index'] for row in comparisons) == [0, 1, 2],
                        'Interleaved Chrome parity is incomplete')
                slot_equal = all(row['slots_equal'] for row in comparisons)
                numeric_equal = all(row['slots_equal'] and row['logits_within_tolerance'] for row in comparisons)
                require(measured['slot_parity_passed'] == slot_equal and measured['numerical_parity_passed'] == numeric_equal,
                        'Interleaved Chrome model parity summary differs')
                engine_checks[name] = {'native_reference_slots_equal': slot_equal,
                    'native_reference_logits_within_tolerance': all(row['logits_within_tolerance'] for row in comparisons),
                    'maximum_absolute_logit_difference': max(row['max_abs'] for row in comparisons)}
        stats[engine] = engine_stats
        checks[engine] = engine_checks
        protocols[engine] = {**protocol, 'source_input_indices': indices, 'input_sha256': input_hashes,
                             'status': report['status']}
        require(report['tolerance'] == {'atol': fixture['atol'], 'rtol': fixture['rtol']}, 'Interleaved parity tolerance changed')
        numerical_passed = all(row['native_reference_slots_equal'] and row['native_reference_logits_within_tolerance']
                               for row in engine_checks.values())
        require(report['numerical_parity_passed'] == numerical_passed
                and (report['status'] == 'passed') == numerical_passed
                and report['slot_parity_passed'] == all(row['native_reference_slots_equal'] for row in engine_checks.values()),
                'Interleaved output parity status differs')
        protocols[engine]['tolerance'] = report['tolerance']
        if engine == 'chrome':
            require(len(report['parity']) == 3 * len(models), 'Unexpected interleaved Chrome parity rows')
            protocols[engine].update(browser=report['version'], execution_context=report['execution_context'],
                                     timer_includes=report['timer_includes'], timer_excludes=report['timer_excludes'])
            environments[engine]['browser'] = report['version']
        identities[engine] = {'report_sha256': sha(report_path.read_bytes()), 'sources_sha256': report['sources_sha256']}
        if report['status'] != 'passed':
            warnings.append({'scope': 'interleaved-latency', 'engine': engine, 'status': report['status'],
                             'note': 'Timing report retains numerical differences; full-corpus native parity is reported per model.'})
    require(hosts['native'] == hosts['chrome'], 'Native and Chrome primary latencies were measured on different hosts')
    return stats, protocols, {'fixture_sha256': sha(subset_path.read_bytes()), 'reports': identities}, warnings, subset, checks, environments


def consolidate(run):
    fixture_file = run/'fixtures.private.json'
    fixture, inventory, upstream, audit = (read(run/name) for name in
        ('fixtures.private.json', 'models.json', 'upstream.json', 'corpus-audit.json'))
    fixture_sha = sha(fixture_file.read_bytes())
    cases, models = fixture['cases'], fixture['models']
    require(fixture['format'] == 'unaveraged-digit-evaluation-fixtures/v1' and len(cases) == fixture['samples'] == SAMPLES,
            'Exactly 2000 evaluation cases are required')
    require(len(models) == MODEL_COUNT and set(models) == set(inventory['models']) and not inventory['missing_models'],
            'Expected all 13 distinct previously tested models')
    require(len({item['id'] for item in cases}) == SAMPLES, 'Duplicate case IDs')
    require(audit['status'] == 'passed' and audit['samples'] == SAMPLES and audit['fixture_sha256'] == fixture_sha,
            'Corpus audit does not identify the final fixture')
    require(audit['model_manifest_sha256'] == sha((run/'models.json').read_bytes()), 'Corpus audit model inventory differs')
    actual_fingerprints = {
        'ordered_inputs_sha256': digest_order(item['sha256'] for item in cases),
        'ordered_labels_sha256': sha(b''.join(json.dumps(item['record'], sort_keys=True).encode() for item in cases)),
        'ordered_image_files_sha256': digest_order(item['source_image_sha256'] for item in cases),
        'ordered_annotation_files_sha256': digest_order(item['source_annotation_sha256'] for item in cases)}
    require(all(fixture[key] == audit[key] == actual_fingerprints[key] for key in FINGERPRINTS), 'Ordered corpus fingerprint differs')
    unique_images = len({item['source_image_sha256'] for item in cases})
    unique_inputs = len({item['sha256'] for item in cases})
    require(unique_images == audit['unique_source_image_hashes'] == 1986
            and unique_inputs == audit['unique_preprocessed_input_hashes'] == 1986, 'Unique image count differs')
    native_runtime, wasm_runtime = verify_upstream(run, upstream)
    split_audit, split_audit_hash = public_split_audit(run, fixture, models)
    schedule_path = run/'accuracy-schedule.private.json'
    schedule = read(schedule_path)
    require(schedule['format'] == 'unaveraged-digit-accuracy-schedule/v1'
            and schedule['chrome']['initial_serial_models'] == 10
            and schedule['chrome']['parallel_final_models'] == list(models)[10:],
            'Accuracy execution schedule differs from the measured model set')
    chrome_schedule = {key: schedule['chrome'][key] for key in
        ('initial_serial_models', 'parallel_final_models', 'overlap', 'handoff', 'latency_scope')}
    timed, timing_protocol, timing_evidence, timing_warnings, latency_subset, timing_checks, latency_environments = interleaved_latency(
        run, fixture, fixture_sha, models, native_runtime, wasm_runtime)
    router_file = ROOT/'.cache/digit/huggingface'/fixture['model_revision']/'question_router.py'
    require(sha(router_file.read_bytes()) == fixture['question_router_sha256'], 'Pinned native question router changed')
    spec = importlib.util.spec_from_file_location('digit_report_question_router', router_file)
    router = importlib.util.module_from_spec(spec); spec.loader.exec_module(router)
    rows, warnings, report_hashes = [], [], {}
    native_protocol = chrome_protocol = environment = None
    native_caller_hashes, chrome_source_hashes = set(), None
    for name, model in models.items():
        require(re.fullmatch(r'[a-zA-Z0-9_-]+', name), 'Unsafe model alias')
        manifest = inventory['models'][name]
        require({key: value for key, value in model.items() if key != 'references_file'} == manifest,
                'Model fixture differs from inventory: ' + name)
        require(set(model['sha256']) == {'graph.json', 'model.safetensors'}, 'Model graph/weights identity required')
        directory = Path(model['directory'])
        require(all(sha((directory/file).read_bytes()) == expected for file, expected in model['sha256'].items()), 'Frozen model bytes changed')
        require(sum((directory/file).stat().st_size for file in model['sha256']) == model['graph_weight_bytes'], 'Model size changed')
        native_path, chrome_path = run/'native'/(name+'.json'), run/'wasm'/name/'evaluation.json'
        native, chrome = read(native_path), read(chrome_path)
        require(native['status'] == 'passed' and native['samples'] == SAMPLES, 'Incomplete native evaluation: ' + name)
        require(chrome['status'] in ('passed', 'completed-with-numerical-differences')
                and chrome['samples'] == chrome['completed'] == SAMPLES, 'Incomplete Chrome evaluation: ' + name)
        require(native['model'] == chrome['model'] == name and native['sha256'] == chrome['model_sha256'] == model['sha256'],
                'Measured model identity differs: ' + name)
        require(native['identity']['fixture_sha256'] == chrome['fixture_sha256'] == fixture_sha, 'Measured fixture identity differs')
        require(native['runtime']['version'] == chrome['runtime']['version'] == upstream['version']
                and native['runtime']['sha256'] == native_runtime['sha256']
                and chrome['runtime']['sha256'] == wasm_runtime['sha256']
                and chrome['native_reference_runtime'] == native['runtime'], 'Measured runtime differs from published artifacts')
        require(chrome['runtime']['source'] == 'published-npm' and native['runtime']['package'] == 'volvoxai', 'Measured runtime source differs')
        require(native['identity']['model_sha256'] == model['sha256']
                and native['identity']['packed_arithmetic'] == chrome['packed_arithmetic'] == model['packed_arithmetic'], 'Measured arithmetic differs')
        require(all(native['evaluation'][key] == fixture[key] for key in FINGERPRINTS)
                and chrome['ordered_inputs_sha256'] == fixture['ordered_inputs_sha256'], 'Measured corpus differs')
        require(native['graph_weight_bytes'] == chrome['model_identity']['graph_weight_bytes'] == model['graph_weight_bytes'], 'Measured model size differs')
        native_caller_hashes.add(native['identity']['caller_sha256'])
        if chrome_source_hashes is None:
            chrome_source_hashes = chrome['sources_sha256']
        require(chrome['sources_sha256'] == chrome_source_hashes, 'Chrome caller changed between models')
        for source, expected in chrome['sources_sha256'].items():
            require(sha((ROOT/source).read_bytes()) == expected, 'Chrome evaluation source changed after measurement')
        refs_path = safe_evidence_path(run/'native', native['references']['file'])
        samples_path = safe_evidence_path(run/'native', native['private_samples']['file'])
        require(sha(refs_path.read_bytes()) == native['references']['sha256'] == chrome['references_sha256']
                and sha(samples_path.read_bytes()) == native['private_samples']['sha256'], 'Private native evidence changed')
        refs, samples = read(refs_path), read(samples_path)
        require(refs['identity'] == samples['identity'] == native['identity'] and refs['status'] == 'passed'
                and len(refs['cases']) == len(samples['cases']) == SAMPLES, 'Incomplete native evidence')
        require(latency_subset['source_references_sha256'][name] == native['references']['sha256'], 'Latency reference source differs')
        require(all(subset_item['references'][name]['sha256'] == refs['cases'][index]['sha256']
                    for subset_item, index in zip(latency_subset['cases'], [0, 999, 1999])), 'Latency reference outputs differ')
        predictions_path = run/'wasm'/name/'predictions.private.jsonl'
        predictions_bytes = predictions_path.read_bytes()
        require(predictions_bytes.endswith(b'\n'), 'Incomplete Chrome checkpoint tail')
        predictions = [json.loads(line) for line in predictions_bytes.splitlines()]
        require(len(predictions) == SAMPLES, 'Incomplete Chrome private records')
        native_counts = dict.fromkeys(SCORE_KEYS, 0); chrome_counts = dict.fromkeys(SCORE_KEYS, 0)
        native_timings, chrome_timings = [], []
        parity = {'slot_mismatches': 0, 'numerical_mismatches': 0, 'byte_mismatches': 0, 'maximum_absolute_logit_difference': 0.0}
        for index, (item, reference, sample, prediction) in enumerate(zip(cases, refs['cases'], samples['cases'], predictions)):
            require(item['id'] == reference['id'] == sample['id'] == prediction['id']
                    and prediction['index'] == index and prediction['model'] == name
                    and item['sha256'] == sample['input_sha256'] == prediction['input_sha256'], 'Private record ordering/identity differs')
            native_record = reference_record(reference)
            require(native_record == sample['record'], 'Native saved prediction differs from logits')
            native_good = native_record == item['record']
            native_answer_good = str(router.answer_from_record(item['question'], **native_record)) == item['answer']
            require(sample['record_correct'] == native_good and sample['answer_correct'] == native_answer_good, 'Native saved scores differ')
            native_counts['record_correct'] += native_good
            native_counts['phone_correct'] += native_record['phone'] == item['record']['phone']
            native_counts['street_correct'] += native_record['street'] == item['record']['street']
            native_counts['answer_correct'] += native_answer_good
            chrome_record = decode_slots(prediction['slots'])
            require(chrome_record == prediction['record'], 'Chrome slots/record differ')
            for metric, expected in (('record_correct', chrome_record == item['record']),
                    ('phone_correct', chrome_record['phone'] == item['record']['phone']),
                    ('street_correct', chrome_record['street'] == item['record']['street'])):
                require(prediction['scores'][metric] == expected, 'Chrome saved field score differs')
                chrome_counts[metric] += expected
            require(len(prediction['answers']) == 1, 'Expected one question per heldout record')
            answer = prediction['answers'][0]
            require(answer['correct'] == (answer['answer'] == item['answer']), 'Chrome saved answer score differs')
            chrome_counts['answer_correct'] += answer['correct']
            comparison = prediction['native']
            require(comparison['slots_equal'] == (prediction['slots'] == reference['slots'])
                    and comparison['bytes_equal'] == (prediction['output_sha256'] == reference['sha256'])
                    and comparison['logits_within_tolerance'] == (comparison['outside_tolerance'] == 0), 'Chrome native comparison differs')
            parity['slot_mismatches'] += not comparison['slots_equal']
            parity['numerical_mismatches'] += not comparison['logits_within_tolerance']
            parity['byte_mismatches'] += not comparison['bytes_equal']
            delta = comparison['max_absolute_logit_difference']
            require(math.isfinite(delta) and delta >= 0, 'Invalid Chrome logit delta')
            parity['maximum_absolute_logit_difference'] = max(parity['maximum_absolute_logit_difference'], delta)
            native_timings.append(sample['inference_ms']); chrome_timings.append(prediction['inference_ms'])
        require(native_counts == native['counts'], 'Native aggregate accuracy differs from private evidence')
        for metric, count in chrome_counts.items():
            require(chrome['metrics'][metric] == {'correct': count, 'count': SAMPLES, 'percent': 100*count/SAMPLES},
                    'Chrome aggregate accuracy differs from private evidence')
        require(chrome['native_parity'] == parity, 'Chrome aggregate parity differs from private evidence')
        native_stats, chrome_stats = latency_stats(native_timings), latency_stats(chrome_timings)
        same_stats(native['latency'], native_stats, name + ' native'); same_stats(chrome['latency'], chrome_stats, name + ' Chrome')
        native_acc, chrome_acc = accuracy(native_counts), accuracy(chrome_counts)
        require(all(math.isclose(native['accuracy_percent'][key], 100*count/SAMPLES, abs_tol=1e-10)
                    for key, count in native_counts.items()), 'Native accuracy percentage differs')
        drift = bool(parity['slot_mismatches'] or parity['numerical_mismatches'])
        require((chrome['status'] == 'completed-with-numerical-differences') == drift, 'Chrome drift status differs')
        if drift:
            warnings.append({'model': name, 'status': chrome['status'], **parity})
        conditions = native['conditions']
        protocol = {key: chrome[key] for key in ('backend', 'cpu_threads', 'batch_size', 'warmups', 'execution_context',
                    'latency_scope', 'timer_includes', 'timer_excludes', 'host_idle_claimed', 'browser', 'tolerance')}
        protocol.update(samples_per_model=SAMPLES, repeats_per_input=1,
                        order='The same ordered heldout inputs once, sequentially within each model. The first ten models use one sequential driver; the final three model processes start concurrently while model ten completes.',
                        execution_schedule=chrome_schedule)
        if native_protocol is None:
            native_protocol, chrome_protocol, environment = conditions, protocol, native['environment']
        require(conditions == native_protocol and protocol == chrome_protocol and native['environment'] == environment,
                'Timing protocol/environment changed between models')
        require(conditions['cpu_threads'] == protocol['cpu_threads'] == 1
                and conditions['batch_size'] == protocol['batch_size'] == 1 and conditions['samples_per_model'] == SAMPLES,
                'Unexpected batch/thread/sample conditions')
        rows.append({'id': name, 'method': model['method'], 'conv_weight_bits': model['conv_weight_bits'],
            'linear_weight_bits': model['linear_weight_bits'], 'activation_policy': model['activation_policy'],
            'packed_arithmetic': model['packed_arithmetic'], 'graph_weight_bytes': model['graph_weight_bytes'],
            'selected': name == SELECTED, 'sha256': model['sha256'],
            'source_exporter_version': model['source_exporter_version'], 'retrained_for_target_runtime': False,
            'accuracy': {'native': native_acc, 'chrome': chrome_acc},
            'latency': {'native': timed['native'][name], 'chrome': timed['chrome'][name]},
            'latency_validation': {engine: timing_checks[engine][name] for engine in ('native', 'chrome')},
            'evaluation_pass_latency': {'native': native_stats, 'chrome': chrome_stats}, 'native_parity': parity,
            'status': {'native': native['status'], 'chrome': chrome['status']}})
        report_hashes[name] = {'native_report_sha256': sha(native_path.read_bytes()), 'chrome_report_sha256': sha(chrome_path.read_bytes()),
            'native_references_sha256': native['references']['sha256'], 'native_samples_sha256': native['private_samples']['sha256'],
            'chrome_predictions_sha256': sha(predictions_bytes)}
    require(len(native_caller_hashes) == 1 and sha((ROOT/'benchmarks/digit_native_eval.py').read_bytes()) in native_caller_hashes,
            'Native evaluation source changed')
    require(sum(row['selected'] for row in rows) == 1, 'Exactly one existing selected model must be marked')
    warnings.extend(timing_warnings)
    report = {'format': 'unaveraged-digit-comparison/v1', 'updated': datetime.now(timezone.utc).date().isoformat(),
        'status': 'completed-with-numerical-differences' if warnings else 'passed', 'aggregate_only': True,
        'runtime': {'native': native_runtime, 'wasm': wasm_runtime}, 'environment': environment,
        'environment_scope': 'Full 2000-record accuracy passes only. Primary timing hosts are recorded separately in latency_environment.',
        'latency_environment': latency_environments,
        'corpus': {'samples': SAMPLES, 'unique_images': unique_images, 'unique_preprocessed_inputs': unique_inputs,
            'split': 'heldout', **actual_fingerprints, 'annotation_caveat': fixture['annotation_caveat'],
            'question_answers_consistent_with_numeric_records': audit['question_answers_consistent_with_numeric_records'],
            'prior_evaluation_use': audit['matches_prior_evaluation_inputs'], 'independent': False,
            'training_performed': False, 'w8a8_cached_calibration_overlap': audit['w8a8_cached_calibration_overlap']},
        'protocol': {'accuracy': {'native': native_protocol, 'chrome': chrome_protocol}, 'latency': timing_protocol,
            'comparison_scope': 'Primary native and Chrome model latency uses the same host, three heldout inputs, 15 cycles and alternating model order. Full 2000-input accuracy-pass latency from the separate accuracy host is retained separately and includes concurrent final-model evaluation; no cross-host speedup is inferred.'},
        'models': rows, 'warnings': warnings,
        'split_audit': split_audit,
        'selection_scope': 'Existing compact default is marked; this evaluation does not retrain models or select a new checkpoint.',
        'scoring': 'Record accuracy requires exact complete phone and street digits. Question accuracy counts each supplied question separately.',
        'evidence': {'fixture_sha256': fixture_sha, 'model_inventory_sha256': sha((run/'models.json').read_bytes()),
            'corpus_audit_sha256': sha((run/'corpus-audit.json').read_bytes()), 'upstream_manifest_sha256': sha((run/'upstream.json').read_bytes()),
            'split_audit_sha256': split_audit_hash,
            'accuracy_schedule_sha256': sha(schedule_path.read_bytes()),
            'native_evaluator_sha256': next(iter(native_caller_hashes)), 'chrome_sources_sha256': chrome_source_hashes,
            'interleaved_latency': timing_evidence,
            'aggregator_sha256': sha(Path(__file__).read_bytes()), 'reports': report_hashes}}
    serialized = json.dumps(report, ensure_ascii=False, allow_nan=False)
    require('/home/' not in serialized and 'predictions.private' not in serialized and '"cases"' not in serialized,
            'Public aggregate contains private paths/records')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = consolidate(args.run_dir.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.out)
    print('DIGIT_REPORT', report['status'], len(report['models']), 'models x', report['corpus']['samples'], 'samples')


if __name__ == '__main__':
    main()
