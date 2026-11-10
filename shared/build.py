#!/usr/bin/env python3
"""Build Chrome and Firefox packages from pinned HF models and public synthetic receipts."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.cpu_policy import resolve_cpu_threads
PROJECTS = ('digit', 'vqa')
BROWSERS = ('chrome', 'firefox')
RUNTIME_VERSION = json.loads((ROOT / 'package.json').read_text())['config']['volvoxai_version']
PRECISIONS = ('w8a8',)
FROZEN_DIGIT_PRECISIONS = ('conv8-linear4', 'conv4-linear4', 'conv4-linear3', 'conv4-linear2')
SUPPORTED_PRECISIONS = (*PRECISIONS, 'w4a8', *FROZEN_DIGIT_PRECISIONS)
EXTENSION_FILES = (
    'background.js', 'content.js', 'content.css', 'manifest.json',
    'offscreen.html', 'offscreen.js', 'popup.html', 'popup.js', 'popup.css', 'heatmap-view.js',
    'firefox-background.html', 'firefox-background.js',
    'settings-core.js', 'settings.js', 'options.html', 'options.js', 'options.css',
    'inference-pool.js',
    'reader.worker.js', 'requests.js', 'runtime.js', 'preprocess.js',
)
SOURCE_FILES = (
    '.gitignore', 'Makefile', 'README.md', 'LICENSE', 'NOTICE', 'PRIVACY.md', 'STORE.md',
    'package.json', 'package-lock.json', 'requirements.txt',
    'shared/build.py', 'shared/release.py', 'shared/docs.py', 'shared/pages.py', 'shared/model_support.py', 'shared/cpu_policy.py', 'shared/sites.json',
    '.github/workflows/pages.yml',
    *(f'example/{name}' for name in ('index.html', 'app.js', 'worker.js', 'style.css', 'browser.mjs', 'questions.js')),
    'docs/style.css', 'docs/requirements.txt',
    *(f'docs/assets/examples/{name}' for name in (
        'digit-receipt-ko.jpg', 'vqa-receipt-ko.jpg', 'sources.json',
    )),
    *(f'synth/{name}' for name in (
        'README.md', 'manifest.json', 'SHA256SUMS',
    )),
    *(f'synth/{folder}/{index:06d}.{extension}'
      for folder, extension in (('images', 'png'), ('annotations', 'json'))
      for index in range(256)),
    'shared/tests/core.test.mjs', 'shared/tests/browser.mjs', 'shared/tests/runtime.test.mjs', 'shared/tests/build.test.mjs',
    'shared/tests/firefox-browser.mjs', 'shared/tests/firefox-runtime.test.mjs',
    'shared/tests/settings.test.mjs', 'shared/tests/browser-settings.mjs', 'shared/tests/heatmap-view.test.mjs', 'shared/tests/example-questions.test.mjs',
    'shared/tests/vqa-profile.test.mjs',
    'shared/tests/digit-evaluation.test.mjs',
    'shared/tests/test_quantization.py', 'shared/tests/test_frozen.py', 'shared/tests/test_batching.py',
    'shared/tests/test_release.py', 'shared/tests/test_pages.py',
    'shared/tests/test_make.py', 'shared/tests/test_cpu_policy.py', 'shared/tests/test_vqa_check.py',
    'shared/tests/test_source_builds.py', 'shared/tests/test_digit_rebuild.py',
    *(f'shared/tests/{name}' for name in (
        'test_vqa_data.py', 'test_vqa_lowbit.py', 'test_vqa_frozen.py', 'test_vqa_fixtures.py', 'test_docs.py',
    )),
    *(f'shared/assets/icons/icon{size}.png' for size in (16, 32, 48, 128)),
    *(f'shared/extension/{name}' for name in EXTENSION_FILES),
    *(f'{project}/{name}' for project in PROJECTS for name in
      ('extension.json', 'README.md', 'adapter.js', 'model.py', 'model.test.mjs')),
    'vqa/tokenizer.js', 'vqa/check.py', 'vqa/frozen.py', 'vqa/frozen-profiles.json',
    'digit/check.py', 'digit/frozen.py', 'digit/frozen-profiles.json', 'digit/qat.py', 'digit/rebuild.py',
    *(f'benchmarks/{name}' for name in (
        'README.md', 'results.json', 'digit-comparison.json', 'native.py', 'browser.mjs', 'runtime-benchmark.js',
        'worker.js',
        'digit_native_eval.py', 'digit_wasm_eval.mjs', 'digit_report.py', 'digit-evaluation.js', 'digit-evaluation-worker.js',
        'digit_latency.py', 'digit_latency.mjs',
        'models.py', 'vqa_data.py', 'vqa_lowbit.py', 'vqa_fixtures.py', 'vqa_wasm.mjs',
        'vqa-wasm-worker.js', 'tables.py',
    )),

)
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}


def project_config(project):
    if project not in PROJECTS:
        raise ValueError(f'Unknown extension: {project}')
    config = json.loads((ROOT / project / 'extension.json').read_text())
    if config['id'] != project:
        raise ValueError('Extension ID differs from its directory')
    return config


def site_config(config, dev=False, *, root=None):
    root = ROOT if root is None else root
    sites = json.loads((root/'shared/sites.json').read_text())
    development = sites.pop('development_matches')
    if dev:
        sites['matches'] += development
    sites.update(channel=config['channel'], ui=config['ui'])
    return sites


def extension_manifest(config, sites, dev=False, *, root=None, browser='chrome'):
    if browser not in BROWSERS:
        raise ValueError(f'Unknown browser: {browser}')
    root = ROOT if root is None else root
    manifest = json.loads((root/'shared/extension/manifest.json').read_text())
    manifest.update({name: config[name] for name in ('name', 'version', 'description')})
    manifest['action']['default_title'] = config['ui']['title']
    manifest['permissions'] = list(dict.fromkeys([*manifest.get('permissions', []), 'storage', 'scripting']))
    manifest['optional_host_permissions'] = ['http://*/*', 'https://*/*']
    manifest['options_ui'] = {'page': 'shared/extension/options.html', 'open_in_tab': True}
    manifest['content_scripts'] = [{'matches': sites['matches'],
        'js': ['shared/extension/site-config.js', 'shared/extension/settings-core.js', 'shared/extension/heatmap-view.js', 'shared/extension/content.js'],
        'css': ['shared/extension/content.css'], 'all_frames': True, 'run_at': 'document_idle'}]
    if browser == 'firefox':
        manifest.pop('minimum_chrome_version', None)
        manifest['permissions'] = [permission for permission in manifest.get('permissions', [])
                                   if permission != 'offscreen']
        manifest['background'] = {'page': 'shared/extension/firefox-background.html', 'persistent': False}
        manifest['browser_specific_settings'] = {'gecko': {
            'id': f"{config['slug']}@unaveraged",
            'strict_min_version': '140.0',
            'data_collection_permissions': {'required': ['none']},
        }}
    if dev:
        manifest['name'] += ' (개발용)'
    return manifest


def prepare_environment() -> None:
    """Use stdlib venv; support Debian Python installations without ensurepip."""
    ensurepip = importlib.util.find_spec("ensurepip") is not None
    if not ensurepip and importlib.util.find_spec("pip") is None:
        raise SystemExit("Install Python's venv/ensurepip support (python3-venv on Debian/Ubuntu) "
                         "or pip for the build interpreter, then run make again.")
    venv.EnvBuilder(with_pip=ensurepip, symlinks=True).create(ROOT / ".venv")
    if not ensurepip:
        # Bootstrap only inside this environment; never install to system Python.
        python = ROOT / ".venv/bin/python"
        target = subprocess.check_output([str(python), "-I", "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip()
        subprocess.run([sys.executable, "-I", "-m", "pip", "--isolated", "install",
            "--disable-pip-version-check", "--index-url", "https://pypi.org/simple",
            "--no-deps", "--target", target, "pip==25.3"], check=True)

def zip_directory(directory: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(directory.rglob("*")):
            if file.is_file():
                bundle.write(file, file.relative_to(directory))

def installed_packages() -> tuple[Path, dict]:
    """Accept only the pinned npm lite runtime and published Python wheel."""
    try:
        python_package = importlib.metadata.distribution("volvoxai")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Run make deps to install requirements.txt") from error
    if python_package.version != RUNTIME_VERSION or python_package.read_text("direct_url.json"):
        raise RuntimeError(f"Published PyPI volvoxai=={RUNTIME_VERSION} is required; run make deps")
    if not python_package.locate_file("volvoxai/_export.py").is_file():
        raise RuntimeError("The installed VolvoxAI wheel is missing its exporter")
    npm_root = ROOT / "node_modules/volvoxai-lite"
    if npm_root.is_symlink() or not (npm_root / "package.json").is_file():
        raise RuntimeError("Run npm ci to install volvoxai-lite")
    package = json.loads((npm_root / "package.json").read_text())
    lock = json.loads((ROOT / "package-lock.json").read_text())["packages"]["node_modules/volvoxai-lite"]
    expected_url = f"https://registry.npmjs.org/volvoxai-lite/-/volvoxai-lite-{RUNTIME_VERSION}.tgz"
    if (package.get("name") != "volvoxai-lite" or package.get("version") != RUNTIME_VERSION
            or lock.get("version") != RUNTIME_VERSION or lock.get("resolved") != expected_url
            or not lock.get("integrity")):
        raise RuntimeError(f"Pinned npm volvoxai-lite@{RUNTIME_VERSION} is required; run npm ci")
    return npm_root, {
        "javascript": {"package":"volvoxai-lite", "version":RUNTIME_VERSION, "source":"published",
                       "resolved":lock["resolved"], "integrity":lock["integrity"]},
        "python": {"package":"volvoxai", "version":RUNTIME_VERSION, "source":"published",
                   "exporter_sha256":digest(python_package.locate_file("volvoxai/_export.py")),
                   "native_sha256":{name:digest(python_package.locate_file("volvoxai/"+name))
                       for name in ("libvolvoxai.so", "libsynurang_module_host.so")}},
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_runtime(npm_root: Path, args, dependencies: dict) -> tuple[Path, dict]:
    """Select a local JS/WASM release pair while preserving model provenance."""
    published = npm_root / 'dist' / RUNTIME_VERSION
    requested = getattr(args, 'runtime_dir', None)
    if requested is None:
        return published, dependencies
    if not args.package_only:
        raise ValueError('--runtime-dir requires --package-only; reuse the fixed model artifacts')
    selected = Path(requested).resolve()
    for name in ('volvoxai.lite.min.js', 'volvoxai.lite.wasm'):
        if not (selected / name).is_file():
            raise ValueError(f'Missing local runtime file: {name}')
    same_api = digest(selected / 'volvoxai.lite.min.js') == digest(published / 'volvoxai.lite.min.js')
    metadata = None
    if not same_api:
        header = (selected / 'volvoxai.lite.min.js').read_text().splitlines()[0]
        match = re.fullmatch(r'/\*! volvoxai-browser-provenance-v1 (\{.*\}) \*/', header)
        if not match:
            raise ValueError('Local JavaScript API requires VolvoxAI release provenance')
        metadata = json.loads(match.group(1))
        if (metadata.get('format') != 'volvoxai-browser-provenance/v1'
                or metadata.get('profile') != 'inference'
                or metadata.get('capability') != 'inference'
                or metadata.get('sidecar') != 'volvoxai.lite.wasm'
                or metadata.get('backends') != ['wasm']
                or not re.fullmatch(r'\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?', metadata.get('packageVersion', ''))):
            raise ValueError('Local JavaScript API is not a versioned lite WASM release')
    with (selected / 'volvoxai.lite.wasm').open('rb') as stream:
        if stream.read(8) != b'\x00asm\x01\x00\x00\x00':
            raise ValueError('Local runtime is not a version-one WASM module')
    javascript = ({'package': 'volvoxai-lite', 'version': RUNTIME_VERSION,
                   'source': 'local-wasm-build', 'javascript_api_source': dependencies['javascript']}
                  if same_api else {'package': 'volvoxai-lite', 'version': metadata['packageVersion'],
                                    'source': 'local-dist', 'browser_provenance': metadata})
    return selected, {**dependencies, 'javascript': javascript}


def export_source():
    """Only this explicit list may enter the public monorepo; no private parent files."""
    output = ROOT / 'dist/source'
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.source-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'source'
        for name in SOURCE_FILES:
            source = ROOT / name
            if source.is_symlink() or not source.is_file():
                raise ValueError(f'Expected a regular source file: {name}')
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(stage), output)
    archive = ROOT / 'dist/unaveraged-source.zip'
    zip_directory(output, archive)
    print(f'Public monorepo: {output}\nSource ZIP: {archive}')


def validate_inputs(args):
    if args.variant != 'ptq' or args.package_only or args.fixtures:
        return
    if args.model_source is not None:
        return
    if args.model == 'digit' and args.precision == 'conv4-linear2':
        # Keep preflight dependency-free; the rebuild verifies hashes and labels.
        required = ['manifest.json', 'SHA256SUMS',
                    *(f'{folder}/{i:06d}.{suffix}' for i in range(256)
                      for folder, suffix in (('images', 'png'), ('annotations', 'json')))]
        for name in required:
            if not (ROOT/'synth'/name).is_file():
                raise ValueError(f'Missing public QAT input: synth/{name}')
        return
    if args.calibration_images is None:
        raise ValueError('PTQ requires CALIBRATION_IMAGES=/path/to/images; EVAL_IMAGES and '
                         'CALIBRATION_QUESTIONS are optional. No private dataset is downloaded.')
    if args.calibration_count < 1:
        raise ValueError('CALIBRATION_COUNT must be positive')
    if args.calibration_questions and not args.calibration_questions.is_file():
        raise ValueError('CALIBRATION_QUESTIONS must point to an existing JSONL file')
    for directory in [args.calibration_images, *args.exclude_images]:
        if not directory.is_dir():
            raise ValueError(f'Image directory does not exist: {directory}')
        images = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        if not images:
            raise ValueError(f'No PNG/JPEG/WebP images directly inside: {directory}')
        if directory == args.calibration_images and len(images) < args.calibration_count:
            raise ValueError(f'Need at least {args.calibration_count} calibration images; found {len(images)}')


def selected_builds(args):
    """Resolve per-extension defaults before validating calibration requirements."""
    selections = []
    for project in PROJECTS if args.model == 'all' else (args.model,):
        config = project_config(project)
        precisions = PRECISIONS if args.precision == 'all' else (
            (config['default_precision'] if args.variant == 'ptq' else 'w8a8')
            if args.precision == 'default' else args.precision,)
        for precision in precisions:
            selected = argparse.Namespace(**vars(args))
            selected.model, selected.precision = project, precision
            selected.frozen_default = False
            selected.model_source = getattr(args, f'{project}_model_source', None) or args.model_source
            if precision in FROZEN_DIGIT_PRECISIONS and (project != 'digit' or args.variant != 'ptq'):
                raise ValueError('Frozen candidate profiles require --model digit and --variant ptq')
            if precision == 'w4a8' and project != 'vqa':
                raise ValueError('W4A8 requires --model vqa')
            if args.variant != 'ptq' and args.variant not in config['model']['variants']:
                raise ValueError(f'{project} does not support variant {args.variant}')
            if not args.package_only and not args.fixtures and args.variant == 'ptq':
                frozen = precision in FROZEN_DIGIT_PRECISIONS
                if selected.model_source is None and frozen and precision != 'conv4-linear2':
                    cached = ROOT / '.cache' / project / 'packages' / RUNTIME_VERSION / 'ptq' / precision
                    if (cached / 'manifest.json').is_file():
                        selected.model_source = cached
                    else:
                        raise ValueError(f'{project} {precision} requires MODEL_SOURCE=/path/to/verified/model '
                                         f'(or {project.upper()}_MODEL_SOURCE for a combined build).')
                if selected.model_source is not None and not (frozen or (project == 'vqa' and precision == 'w4a8')):
                    raise ValueError('MODEL_SOURCE is only supported for frozen Digit profiles and VQA W4A8')
            validate_inputs(selected)
            selections.append((config, selected))
    return selections


def privacy_page(*, root=None):
    root = ROOT if root is None else root
    from shared.docs import markdown_html
    body = markdown_html((root/'PRIVACY.md').read_text())
    return ('<!doctype html><html lang="ko"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width"><title>unaveraged — Privacy</title>'
            '<style>body{max-width:760px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;'
            'color:#172d26}h1{font-size:25px}h2{font-size:20px}p{white-space:pre-line}</style>'
            '<main>' + body + '</main></html>\n')


def download(config, name, directory, expected=None):
    target = directory / name
    if expected and target.is_file() and digest(target) == expected:
        return
    model = config['model']
    url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{name}"
    print(f"[{config['id']}] Download {name}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent':'receipt-reader-extensions'})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    if expected and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'SHA256 mismatch for {name}')
    temporary = target.with_suffix(target.suffix + '.part')
    temporary.write_bytes(data)
    temporary.replace(target)


def fetch_release(config, variant, *, fixtures_only=False):
    model = config['model']
    source_variant = 'fp32' if variant == 'ptq' else variant
    if source_variant not in model['variants']:
        raise ValueError(f"{config['id']} does not support variant {variant}")
    source = ROOT / '.cache' / config['id'] / 'huggingface' / model['revision']
    download(config, 'SHA256SUMS', source)
    checksums = {}
    for line in (source / 'SHA256SUMS').read_text().splitlines():
        if line.strip():
            digest_value, name = line.split(maxsplit=1)
            checksums[name.lstrip('*')] = digest_value
    files = model['common_files'] + ([] if fixtures_only else model['variants'][source_variant])
    if any(name not in checksums for name in files):
        raise ValueError('Incomplete release SHA256SUMS')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda name: download(config, name, source, checksums[name]), files))
    return source


def build(config, args):
    project = config['id']
    cpu_threads = resolve_cpu_threads(getattr(args, 'cpu_threads', None))
    module = importlib.import_module(f'{project}.model')
    npm_root, dependencies = installed_packages()
    runtime_dist, dependencies = select_runtime(npm_root, args, dependencies)
    runtime_files = ['volvoxai.lite.min.js', 'volvoxai.lite.wasm']
    for name in runtime_files:
        if not (runtime_dist / name).is_file():
            raise RuntimeError(f'Missing npm runtime: {name}')
    source = fetch_release(config, args.variant)
    packages = ROOT / '.cache' / project / 'packages' / RUNTIME_VERSION
    package = packages / args.variant
    if args.variant == 'ptq':
        package /= args.precision
    package.parent.mkdir(parents=True, exist_ok=True)
    if args.package_only:
        if not (package / 'manifest.json').is_file():
            raise ValueError(f'Build {project} once before --package-only')
        cached = json.loads((package / 'manifest.json').read_text())
        frozen = bool(cached.get('frozen_candidate')) and (
            (project == 'digit' and args.precision in FROZEN_DIGIT_PRECISIONS)
            or (project == 'vqa' and args.precision == 'w4a8'))
        if (not frozen and cached['converter']['version'] != RUNTIME_VERSION) or cached['variant'] != args.variant:
            raise ValueError('Cached package version/variant mismatch')
        if args.variant == 'ptq' and cached.get('quantization', {}).get('precision') != args.precision:
            raise ValueError('Cached package precision mismatch')
        hashes = cached.get('source_sha256') or cached.get('source', {}).get('sha256')
        if not hashes or any(digest(source / name) != expected for name,expected in hashes.items()):
            raise ValueError('Cached package source mismatch')
    else:
        print(f'[{project}] Requested CPU threads: {cpu_threads} (one model job at a time)', flush=True)
        with tempfile.TemporaryDirectory(prefix='.convert-', dir=package.parent) as temporary:
            converted = Path(temporary) / args.variant
            if args.precision in FROZEN_DIGIT_PRECISIONS and args.model_source is not None:
                from digit.frozen import import_model
                import_model(args.model_source, converted, source, args.precision)
            elif project == 'vqa' and args.variant == 'ptq' and getattr(args, 'model_source', None) is not None:
                from vqa.frozen import import_model
                import_model(args.model_source, converted, source, args.precision)
            elif args.variant == 'ptq':
                prepared = Path(temporary) / 'fp32-prepared'
                module.convert_model(source, prepared, 'fp32', RUNTIME_VERSION, prepare_for_ptq=True)
                if project == 'digit' and args.precision == 'conv4-linear2':
                    from digit.rebuild import rebuild_model
                    rebuild_model(source, converted, prepared, synth_root=ROOT/'synth',
                                  backend=getattr(args, 'qat_backend', 'cpu'), cpu_threads=cpu_threads)
                else:
                    module.quantize_model(prepared, converted, source=source,
                        questions=args.calibration_questions if config['cache_by_question'] else None,
                        images=args.calibration_images.resolve(),
                        excluded=[p.resolve() for p in args.exclude_images] + [source/'examples'],
                        count=args.calibration_count, seed=args.calibration_seed,
                        evaluation_checked=bool(args.exclude_images), precision=args.precision,
                        cpu_threads=cpu_threads)
                reference = packages / 'fp32-prepared'
                if reference.exists():
                    shutil.rmtree(reference)
                shutil.move(str(prepared), reference)
            else:
                module.convert_model(source, converted, args.variant, RUNTIME_VERSION)
            if package.exists():
                shutil.rmtree(package)
            shutil.move(str(converted), package)

    manifest = json.loads((package/'manifest.json').read_text())
    if project == 'digit' and manifest.get('frozen_candidate'):
        from digit.frozen import validate_package
        validate_package(package, source, args.precision)
    elif project == 'digit' and args.precision == 'conv4-linear2':
        from digit.rebuild import validate_package
        validate_package(package, source, args.precision)
    if project == 'vqa' and manifest.get('frozen_candidate'):
        from vqa.frozen import validate_package
        validate_package(package, args.precision)

    browser = getattr(args, 'browser', 'chrome')
    if browser not in (*BROWSERS, 'all'):
        raise ValueError(f'Unknown browser: {browser}')
    outputs = []
    # Prepare and validate each model once, then reuse its bytes for both browsers.
    for browser in BROWSERS if browser == 'all' else (browser,):
        output_name = browser + ('-dev' if args.dev else '') + ('-local' if getattr(args, 'runtime_dir', None) else '')
        output = ROOT / 'dist' / project / (args.precision if args.variant == 'ptq' else args.variant) / output_name
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.package-', dir=output.parent) as temporary:
            stage = Path(temporary) / output_name
            shared = stage / 'shared/extension'
            shared.mkdir(parents=True)
            for name in EXTENSION_FILES:
                if name != 'manifest.json':
                    shutil.copy2(ROOT/'shared/extension'/name, shared/name)
            for name in ('LICENSE', 'NOTICE'):
                shutil.copy2(ROOT/name, stage/f'{name}.txt')
            (stage/'privacy.html').write_text(privacy_page())
            assets = stage / 'shared/assets'
            (assets/'icons').mkdir(parents=True)
            for name in (f'icons/icon{size}.png' for size in (16,32,48,128)):
                shutil.copy2(ROOT/'shared/assets'/name, assets/name)
            (stage/'vendor').mkdir()
            for name in runtime_files:
                shutil.copy2(runtime_dist/name, stage/'vendor'/name)
            shutil.copy2(npm_root/'LICENSE', stage/'vendor/LICENSE.txt')
            (stage/'model').mkdir()
            for name in config['scripts']:
                shutil.copy2(ROOT/project/name, stage/'model'/name)
            model_files = ([str(path.relative_to(package)) for path in module.package_model_files(package)]
                           if hasattr(module, 'package_model_files') else config['model']['package_files'])
            for name in model_files:
                (stage/'model'/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(package/name, stage/'model'/name)
            shutil.copy2(source/'LICENSE', stage/'model/LICENSE.txt')
            if 'NOTICE' in config['model']['common_files']:
                shutil.copy2(source/'NOTICE', stage/'model/NOTICE.txt')
            provenance = {
                'extension':project, 'extension_version':config['version'], 'browser':browser,
                'model_repository':config['model']['repository'], 'model_revision':config['model']['revision'],
                'model_variant':args.variant, 'runtime_version':dependencies['javascript']['version'], 'runtime_profile':'lite',
                'model_precision':args.precision if args.variant == 'ptq' else args.variant,
                'dependencies':dependencies,
                'model_quantization':json.loads((package/'manifest.json').read_text()).get('quantization'),
                'runtime_sha256':{name:digest(stage/'vendor'/name) for name in runtime_files},
                'model_sha256':{name:digest(stage/'model'/name) for name in model_files},
            }
            (stage/'build-info.json').write_text(json.dumps(provenance, indent=2)+'\n')
            sites = site_config(config, args.dev)
            (shared/'site-config.js').write_text('globalThis.RECEIPT_READER_SITES = '+json.dumps(sites,ensure_ascii=False)+';\n')
            manifest = extension_manifest(config, sites, args.dev, browser=browser)
            (stage/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
            if output.exists():
                shutil.rmtree(output)
            shutil.move(str(stage), output)
        archive = output.parent / f"{config['slug']}-{output_name}.zip"
        zip_directory(output, archive)
        print(f'[{project}/{browser}] Package: {output}\nZIP: {archive}', flush=True)
        outputs.append(output)
    return outputs[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all',*PROJECTS], default='all')
    parser.add_argument('--browser', choices=['all', *BROWSERS], default='all',
                        help='Package both browsers by default without repeating model conversion')
    parser.add_argument('--variant', choices=['ptq','fp32'], default='ptq')
    parser.add_argument('--precision', choices=['default', 'all', *SUPPORTED_PRECISIONS], default=os.environ.get('PRECISION') or 'default')
    parser.add_argument('--qat-backend', choices=('cpu', 'cuda'), default=os.environ.get('QAT_BACKEND') or 'cpu',
                        help='Digit public-synth QAT backend (CPU by default; CUDA requires an NVIDIA GPU)')
    parser.add_argument('--cpu-threads', default=os.environ.get('JOBS', 'auto'),
                        help='Native model-preparation threads; auto uses available logical CPUs minus two')
    parser.add_argument('--model-source', type=Path, default=os.environ.get('MODEL_SOURCE') or None,
                        help='Directory containing an exact benchmark candidate (single-model builds)')
    for project in PROJECTS:
        parser.add_argument(f'--{project}-model-source', type=Path,
                            default=os.environ.get(f'{project.upper()}_MODEL_SOURCE') or None)
    if 'CALIBRATION_IMAGES' in os.environ:
        raw_calib = os.environ['CALIBRATION_IMAGES'].strip()
        default_calibration_images = Path(raw_calib) if raw_calib else None
    else:
        default_calibration_images = ROOT / 'synth/images'
    default_seed = int(os.environ['CALIBRATION_SEED']) if os.environ.get('CALIBRATION_SEED') else 20261027
    default_count = int(os.environ['CALIBRATION_COUNT']) if os.environ.get('CALIBRATION_COUNT') else 256
    parser.add_argument('--calibration-images', type=Path, default=default_calibration_images)
    parser.add_argument('--calibration-questions', type=Path, default=os.environ.get('CALIBRATION_QUESTIONS') or None)
    parser.add_argument('--calibration-count', type=int, default=default_count)
    parser.add_argument('--calibration-seed', type=int, default=default_seed)
    parser.add_argument('--exclude-images', type=Path, nargs='*',
        default=[Path(os.environ['EVAL_IMAGES'])] if os.environ.get('EVAL_IMAGES') else [])
    parser.add_argument('--package-only', action='store_true', help='Repackage existing model artifacts')
    parser.add_argument('--runtime-dir', type=Path,
                        help='Explicit local lite JS/WASM directory; requires --package-only and writes browser-specific -local packages')
    parser.add_argument('--dev', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--source', action='store_true')
    mode.add_argument('--check-inputs', action='store_true')
    mode.add_argument('--venv', action='store_true')
    mode.add_argument('--fixtures', action='store_true', help='Download public test fixtures without quantization')
    args = parser.parse_args()
    if args.source:
        export_source()
    elif args.venv:
        prepare_environment()
    else:
        try:
            args.cpu_threads = resolve_cpu_threads(args.cpu_threads)
            if args.runtime_dir is not None and not args.package_only:
                raise ValueError('--runtime-dir requires --package-only')
            if args.model == 'all' and args.model_source is not None:
                raise ValueError('Use DIGIT_MODEL_SOURCE and VQA_MODEL_SOURCE for a combined build')
            selections = selected_builds(args)
        except ValueError as error:
            parser.error(str(error))
        if not args.check_inputs:
            for config, selected in selections:
                if selected.fixtures:
                    fetch_release(config, 'ptq', fixtures_only=True)
                    continue
                build(config, selected)


if __name__ == '__main__':
    main()
