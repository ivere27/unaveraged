import test from 'node:test';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const python = fileURLToPath(new URL('../../.venv/bin/python', import.meta.url));
const root = fileURLToPath(new URL('../../', import.meta.url));
const prelude = String.raw`
import argparse, hashlib, importlib.util, json, os, pathlib, shutil, sys, tempfile, types, zipfile
from unittest.mock import patch
root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('extension_build', root/'shared/build.py')
build = importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
version = '0.9.0'
assert build.RUNTIME_VERSION == version
assert build.PRECISIONS == ('w8a8',)

def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path

def dependencies(repository):
    runtime = repository/'node_modules/volvoxai-lite'
    write(runtime/'package.json', json.dumps({'name':'volvoxai-lite','version':version}))
    write(runtime/'LICENSE', 'runtime license')
    for name in ('volvoxai.lite.min.js','volvoxai.lite.wasm'):
        write(runtime/'dist'/version/name, 'published-'+name)
    lock = {'version':version, 'resolved':f'https://registry.npmjs.org/volvoxai-lite/-/volvoxai-lite-{version}.tgz',
            'integrity':'sha512-fixture-package-integrity'}
    write(repository/'package-lock.json', json.dumps({'packages':{'node_modules/volvoxai-lite':lock}}))
    installed = repository/'site-packages'
    for name in ('_export.py','libvolvoxai.so','libsynurang_module_host.so'):
        write(installed/'volvoxai'/name, 'installed-'+name)
    package = types.SimpleNamespace(version=version, locate_file=lambda name:installed/name,
                                   read_text=lambda name:None)
    return runtime, installed, package, lock
`;
function check(script) {
  execFileSync(python, ['-I', '-c', prelude + script, root], { encoding: 'utf8', maxBuffer: 4_000_000 });
}

test('browser manifests use supported backgrounds and Firefox declares no data collection', () => check(String.raw`
ids=[]
for project in build.PROJECTS:
    config=build.project_config(project)
    sites=build.site_config(config)
    chrome=build.extension_manifest(config,sites)
    firefox=build.extension_manifest(config,sites,browser='firefox')
    assert chrome['background']=={'service_worker':'shared/extension/background.js','type':'module'}
    assert chrome['permissions']==['offscreen','storage','scripting']
    assert 'browser_specific_settings' not in chrome
    assert firefox['background']=={'page':'shared/extension/firefox-background.html','persistent':False}
    assert firefox['permissions']==['storage','scripting']
    assert 'minimum_chrome_version' not in firefox
    gecko=firefox['browser_specific_settings']['gecko']
    assert gecko['strict_min_version']=='140.0'
    assert gecko['data_collection_permissions']=={'required':['none']}
    ids.append(gecko['id'])
    for field in ('content_scripts','content_security_policy','action','icons','optional_host_permissions','options_ui'):
        assert firefox[field]==chrome[field]
    assert chrome['optional_host_permissions']==['http://*/*','https://*/*']
    assert chrome['options_ui']=={'page':'shared/extension/options.html','open_in_tab':True}
    assert chrome['content_scripts'][0]['js']==['shared/extension/site-config.js',
        'shared/extension/settings-core.js','shared/extension/heatmap-view.js','shared/extension/content.js']
    assert not any('127.0.0.1' in match for match in firefox['content_scripts'][0]['matches'])
    dev=build.extension_manifest(config,build.site_config(config,True),True,browser='firefox')
    assert any('127.0.0.1' in match for match in dev['content_scripts'][0]['matches'])
assert len(set(ids))==2
for selected in (None,'chrome','firefox'):
    argv=['shared/build.py','--package-only']+(['--browser',selected] if selected else [])
    with patch.object(sys,'argv',argv),patch.object(build,'selected_builds',return_value=[]) as select:
        build.main()
        assert select.call_args.args[0].browser==(selected or 'all')
`));

test('published lite runtime provenance includes the pinned npm integrity and native hashes', () => check(String.raw`
with tempfile.TemporaryDirectory(prefix='unaveraged provenance ') as temporary:
    repository = pathlib.Path(temporary)
    runtime, installed, package, lock = dependencies(repository)
    with patch.object(build,'ROOT',repository), patch.object(build.importlib.metadata,'distribution',return_value=package):
        selected, info = build.installed_packages()
        assert selected == runtime
        assert info['javascript'] == {'package':'volvoxai-lite','version':version,'source':'published',**lock}
        assert info['python']['source'] == 'published'
        for name in ('libvolvoxai.so','libsynurang_module_host.so'):
            assert info['python']['native_sha256'][name] == build.digest(installed/'volvoxai'/name)
        assert info['python']['exporter_sha256'] == build.digest(installed/'volvoxai/_export.py')
`));

test('local Python installs and unpinned npm packages are refused', () => check(String.raw`
with tempfile.TemporaryDirectory(prefix='unaveraged wrong dependency ') as temporary:
    repository = pathlib.Path(temporary)
    runtime, installed, package, lock = dependencies(repository)
    with patch.object(build,'ROOT',repository), patch.object(build.importlib.metadata,'distribution',return_value=package):
        for wrong_version, direct in [('0.7.0',None),(version,'{"url":"file:///local-runtime.whl"}')]:
            package.version = wrong_version
            package.read_text = lambda name:direct
            try: build.installed_packages()
            except RuntimeError: pass
            else: raise AssertionError('A local or mismatched Python package was accepted')
        package.version=version; package.read_text=lambda name:None
        for field,value in [('version','0.7.0'),('resolved','file:local-runtime.tgz'),('integrity','')]:
            altered={**lock,field:value}
            write(repository/'package-lock.json',json.dumps({'packages':{'node_modules/volvoxai-lite':altered}}))
            try: build.installed_packages()
            except RuntimeError: pass
            else: raise AssertionError('An unpinned npm runtime was accepted')
`));

test('local runtime selection requires fixed models and records release version separately', () => check(String.raw`
with tempfile.TemporaryDirectory(prefix='unaveraged local runtime ') as temporary:
    base=pathlib.Path(temporary); runtime, installed, package, lock=dependencies(base)
    local=base/'local'; local.mkdir()
    write(local/'volvoxai.lite.min.js',(runtime/'dist'/version/'volvoxai.lite.min.js').read_bytes())
    write(local/'volvoxai.lite.wasm',b'\0asm\1\0\0\0')
    args=argparse.Namespace(runtime_dir=local,package_only=False)
    dependencies_info={'javascript':{'source':'published','package':'volvoxai-lite','version':version},'python':{'source':'published'}}
    try:build.select_runtime(runtime,args,dependencies_info)
    except ValueError as error:assert '--package-only' in str(error)
    else:raise AssertionError('Local runtime permitted during model generation')
    args.package_only=True
    selected, info=build.select_runtime(runtime,args,dependencies_info)
    assert selected==local
    assert info['javascript']['source']=='local-wasm-build'
    assert info['javascript']['javascript_api_source']==dependencies_info['javascript']
    assert dependencies_info['javascript']['source']=='published'
    write(local/'volvoxai.lite.min.js','different API')
    try:build.select_runtime(runtime,args,dependencies_info)
    except ValueError as error:assert 'JavaScript API' in str(error)
    else:raise AssertionError('Mismatched JS API accepted')
    metadata={'format':'volvoxai-browser-provenance/v1','packageVersion':'0.9.0',
              'profile':'inference','capability':'inference','backends':['wasm'],
              'sidecar':'volvoxai.lite.wasm'}
    write(local/'volvoxai.lite.min.js','/*! volvoxai-browser-provenance-v1 '+json.dumps(metadata)+' */\nexport const newer=true;')
    selected, info=build.select_runtime(runtime,args,dependencies_info)
    assert info['javascript']['version']=='0.9.0'
    assert info['javascript']['source']=='local-dist'
    assert info['javascript']['browser_provenance']==metadata
    assert info['python']==dependencies_info['python']
    metadata['profile']='full'
    write(local/'volvoxai.lite.min.js','/*! volvoxai-browser-provenance-v1 '+json.dumps(metadata)+' */\n')
    try:build.select_runtime(runtime,args,dependencies_info)
    except ValueError as error:assert 'lite WASM release' in str(error)
    else:raise AssertionError('Wrong runtime profile accepted')
`));

test('Chrome and Firefox ZIPs share each model and runtime without repeating conversion', () => check(String.raw`
with tempfile.TemporaryDirectory(prefix='unaveraged packaging ') as temporary:
    base=pathlib.Path(temporary); repository=base/'repository with spaces'
    for name in build.SOURCE_FILES:
        target=repository/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(root/name,target)
    runtime, installed, package, lock=dependencies(repository)
    source=base/'source'
    for name in ('LICENSE','NOTICE','model.onnx','encoder_model.onnx','decoder_model.onnx'):
        write(source/name,'public model '+name)
    calls=[]; active={}
    def convert_model(source,destination,variant,version,**kwargs):
        calls.append('convert')
        for name in active['model']['package_files']:write(destination/name,'{}')
        write(destination/'manifest.json',json.dumps({'variant':variant,'converter':{'version':version},
            'source_sha256':{name:build.digest(source/name) for name in active['model']['variants']['fp32']}}))
    def quantize_model(prepared,destination,*,precision,**kwargs):
        assert precision=='w8a8'; calls.append('quantize'); shutil.copytree(prepared,destination)
        manifest=json.loads((destination/'manifest.json').read_text())
        manifest.update(variant='ptq',quantization={'precision':precision,'packed_arithmetic':'f32'})
        write(destination/'manifest.json',json.dumps(manifest))
        for name in active['model']['package_files']:
            if name.endswith('.safetensors'):write(destination/name,active['id']+name)
        if active['id']=='vqa':write(destination/'shared/token-embedding.safetensors','one shared shard')
    def package_model_files(destination):
        files=[destination/name for name in active['model']['package_files']]
        shared=destination/'shared/token-embedding.safetensors'
        return files+([shared] if shared.exists() else [])
    converter=types.SimpleNamespace(convert_model=convert_model,quantize_model=quantize_model,package_model_files=package_model_files)
    args=argparse.Namespace(variant='ptq',package_only=False,precision='w8a8',dev=False,browser='all',
        calibration_questions=None,calibration_images=base,exclude_images=[],calibration_count=1,calibration_seed=7)
    with patch.object(build,'ROOT',repository), patch.object(build.importlib.metadata,'distribution',return_value=package), \
         patch.object(build.importlib,'import_module',return_value=converter), patch.object(build,'fetch_release',return_value=source):
        for project in ('digit','vqa'):
            active=build.project_config(project)
            output=build.build(active,args)
            assert output==repository/'dist'/project/'w8a8/chrome'
            info=json.loads((output/'build-info.json').read_text())
            assert info['model_precision']=='w8a8'
            assert info['browser']=='chrome'
            assert info['dependencies']['javascript']['package']=='volvoxai-lite'
            archive=output.parent/(active['slug']+'-chrome.zip')
            with zipfile.ZipFile(archive) as bundle:
                assert json.loads(bundle.read('build-info.json'))==info
                assert bundle.read('manifest.json')== (output/'manifest.json').read_bytes()
                for name in ('volvoxai.lite.min.js','volvoxai.lite.wasm'):
                    expected=(runtime/'dist'/version/name).read_bytes()
                    assert bundle.read('vendor/'+name)==expected
                    assert info['runtime_sha256'][name]==hashlib.sha256(expected).hexdigest()
                for name,expected in info['model_sha256'].items():
                    assert hashlib.sha256(bundle.read('model/'+name)).hexdigest()==expected
                if project=='vqa':assert bundle.namelist().count('model/shared/token-embedding.safetensors')==1
            firefox_output=output.parent/'firefox'
            firefox_info=json.loads((firefox_output/'build-info.json').read_text())
            assert firefox_info=={**info,'browser':'firefox'}
            firefox_archive=output.parent/(active['slug']+'-firefox.zip')
            with zipfile.ZipFile(archive) as chrome_bundle,zipfile.ZipFile(firefox_archive) as firefox_bundle:
                assert set(chrome_bundle.namelist())==set(firefox_bundle.namelist())
                for name in chrome_bundle.namelist():
                    if name not in ('manifest.json','build-info.json'):
                        assert chrome_bundle.read(name)==firefox_bundle.read(name),name
                firefox_manifest=json.loads(firefox_bundle.read('manifest.json'))
                assert 'service_worker' not in firefox_manifest['background']
                assert firefox_manifest['background']['page'] in firefox_bundle.namelist()
            args.package_only=True
            build.build(active,args)
            published_archive_hash=build.digest(archive)
            args.browser='firefox'
            assert build.build(active,args)==firefox_output
            assert build.digest(archive)==published_archive_hash
            args.browser='all'
            local=base/'local-runtime'
            metadata={'format':'volvoxai-browser-provenance/v1','packageVersion':'0.9.0',
                      'profile':'inference','capability':'inference','backends':['wasm'],
                      'sidecar':'volvoxai.lite.wasm'}
            write(local/'volvoxai.lite.min.js','/*! volvoxai-browser-provenance-v1 '+json.dumps(metadata)+' */\n')
            write(local/'volvoxai.lite.wasm',b'\0asm\1\0\0\0')
            args.runtime_dir=local
            local_output=build.build(active,args)
            assert local_output.name=='chrome-local'
            local_info=json.loads((local_output/'build-info.json').read_text())
            assert local_info['dependencies']['javascript']['source']=='local-dist'
            assert local_info['runtime_version']=='0.9.0'
            assert local_info['dependencies']['python']['version']==version
            assert local_info['runtime_sha256']['volvoxai.lite.wasm']==build.digest(local/'volvoxai.lite.wasm')
            assert build.digest(archive)==published_archive_hash
            assert (local_output/'vendor/volvoxai.lite.wasm').read_bytes()==(local/'volvoxai.lite.wasm').read_bytes()
            args.runtime_dir=None
            cached=repository/'.cache'/project/'packages'/version/'ptq/w8a8/manifest.json'
            manifest=json.loads(cached.read_text());manifest['quantization']['precision']='w4a8';cached.write_text(json.dumps(manifest))
            before=build.digest(archive)
            try:build.build(active,args)
            except ValueError as error:assert 'precision mismatch' in str(error)
            else:raise AssertionError('A mislabeled precision was packaged')
            assert build.digest(archive)==before
            args.package_only=False
        assert len(calls)==4,'Package-only must reuse existing models'
`));

test('runtime upgrade accepts exact frozen exports and rejects stale or relabeled packages', () => check(String.raw`
from digit import frozen
with tempfile.TemporaryDirectory(prefix='unaveraged frozen runtime upgrade ') as temporary:
    base=pathlib.Path(temporary); repository=base/'repository'
    for name in build.SOURCE_FILES:
        target=repository/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(root/name,target)
    runtime, installed, package, lock=dependencies(repository)
    source=base/'source'
    for name in ('LICENSE','NOTICE','manifest.json','model.onnx'):
        write(source/name,'public fixture '+name)
    write(source/'config.json',json.dumps({'slots':16,'phone_slots':12,'street_slots':4,
        'num_classes':11,'blank_class':10,'input':{'width':672,'height':320}}))
    precision='conv4-linear2'
    profile=frozen.profiles()[precision]
    original=base/'frozen-original'
    graph={'inputs':{'input0':{'shape':[1,1,320,672],'dtype':'float32'}},'outputs':['slot_logits'],
           'nodes':[{'opType':'QConv2D','params':{'weight_bits':4}} for _ in range(11)]+
                   [{'opType':'PackedLinear','params':{'weight_bits':2,'group_size':32}} for _ in range(7)]}
    write(original/'graph.json',json.dumps(graph)); write(original/'model.safetensors',b'frozen fixture weights')
    profile['sha256']={name:build.digest(original/name) for name in ('graph.json','model.safetensors')}
    cached=repository/'.cache/digit/packages'/version/'ptq'/precision
    converter=types.SimpleNamespace(package_model_files=lambda directory:
        [directory/name for name in ('graph.json','model.safetensors','manifest.json')])
    args=argparse.Namespace(variant='ptq',package_only=True,precision=precision,dev=False,frozen_default=True)
    with patch.object(frozen,'profiles',return_value={precision:profile}):
        frozen.import_model(original,cached,source,precision)
        original_manifest=(cached/'manifest.json').read_bytes()
        assert json.loads(original_manifest)['converter']['version']=='0.8.0'
        with patch.object(build,'ROOT',repository), patch.object(build.importlib.metadata,'distribution',return_value=package), \
             patch.object(build.importlib,'import_module',return_value=converter), patch.object(build,'fetch_release',return_value=source):
            config=build.project_config('digit'); output=build.build(config,args)
            info=json.loads((output/'build-info.json').read_text())
            assert info['runtime_version']==version=='0.9.0'
            assert info['model_quantization']['version']=='0.8.0'
            manifest_path = (output/'model/model-manifest.json') if (output/'model/model-manifest.json').exists() else (output/'model/manifest.json')
            assert manifest_path.read_bytes()==original_manifest
            archive=output.parent/(config['slug']+'-chrome.zip'); expected_archive=build.digest(archive)
            for mutate in (lambda m:m['converter'].update(version=version),
                           lambda m:m['quantization'].update(packed_arithmetic='f32'),
                           lambda m:m.update(private_records=['synthetic private fixture'])):
                altered=json.loads(original_manifest); mutate(altered)
                write(cached/'manifest.json',json.dumps(altered))
                try:build.build(config,args)
                except ValueError as error:assert 'manifest differs' in str(error)
                else:raise AssertionError('Modified frozen manifest was accepted')
                assert build.digest(archive)==expected_archive
            write(cached/'manifest.json',original_manifest)
            args.precision='w8a8'; args.frozen_default=False
            stale=cached.parent/'w8a8'
            for marker in ({},{'frozen_candidate':profile}):
                write(stale/'manifest.json',json.dumps({'variant':'ptq','converter':{'version':'0.8.0'},
                    'quantization':{'precision':'w8a8'},**marker}))
                try:build.build(config,args)
                except ValueError as error:assert 'version/variant mismatch' in str(error)
                else:raise AssertionError('An ordinary stale package bypassed the runtime version check')
`));
