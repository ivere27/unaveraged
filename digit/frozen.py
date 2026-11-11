"""Import and verify immutable candidate weights without fitting new weights."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def profiles():
    return json.loads((ROOT/'digit/frozen-profiles.json').read_text())['profiles']


def validate_graph(graph, profile):
    if graph.get('inputs') != {'input0': {'shape':[1,1,320,672], 'dtype':'float32'}}:
        raise ValueError('Frozen digit input contract changed')
    if graph.get('outputs') != ['slot_logits']:
        raise ValueError('Frozen digit output contract changed')
    convolution = [n for n in graph['nodes'] if n['opType']=='QConv2D']
    dense = [n for n in graph['nodes'] if n['opType']=='PackedLinear']
    if len(convolution)!=11 or len(dense)!=7 or any(n['opType']=='Conv2D' for n in graph['nodes']):
        raise ValueError('Frozen digit operator contract changed')
    if any(n['params'].get('weight_bits',8)!=profile['conv_weight_bits'] for n in convolution):
        raise ValueError('Frozen convolution precision changed')
    if any((n['params'].get('weight_bits'),n['params'].get('group_size')) !=
           (profile['linear_weight_bits'],profile['linear_group_size']) for n in dense):
        raise ValueError('Frozen linear precision changed')


def _verify_model(directory, profile):
    configured = json.loads((ROOT/'digit/extension.json').read_text())['model']
    if (configured['repository'], configured['revision']) != (profile['model_repository'], profile['model_revision']):
        raise ValueError('Frozen candidate belongs to a different source-model revision')
    for name, expected in profile['sha256'].items():
        path = directory/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen candidate hash mismatch: {name}')
    validate_graph(json.loads((directory/'graph.json').read_text()), profile)


def _manifest(release, profile, precision):
    # Preserve released model provenance and ABI without local experiment paths.
    from digit.model import PACKAGE_FORMAT, SOURCE_FORMAT, INPUT_SHAPE
    config = json.loads((release/'config.json').read_text())
    return {'format':PACKAGE_FORMAT,'variant':'ptq','source_format':SOURCE_FORMAT,
        'source':{'sha256':{name:hashlib.sha256((release/name).read_bytes()).hexdigest()
                  for name in ('config.json','manifest.json','model.onnx')}},
        'converter':{'package':'volvoxai','version':'0.8.0','source':'frozen-candidate-import'},
        'abi':{'input':{'name':'input0','shape':INPUT_SHAPE},'output':'slot_logits'},
        'decode':{key:config[key] for key in ('slots','phone_slots','street_slots','num_classes','blank_class')},
        'preprocess':config['input'],
        'quantization':{'method':profile.get('method', 'volvoxai-ptq'),'version':'0.8.0','source_variant':'fp32',
            'precision':precision,'activation_dtype':'int8','packed_arithmetic':'dynamic-i8',
            'conv_weight_bits':profile['conv_weight_bits'],'linear_weight_bits':profile['linear_weight_bits'],
            'group_size':profile['linear_group_size'],'conv_algorithm':profile['conv_algorithm'],
            'linear_algorithm':profile['linear_algorithm'],'quantized_convolutions':11,
            'packed_linear_nodes':7,'quantized_nodes':18,
            'scope':'Static INT8 activations for convolutions; dynamic INT8 activations for packed linear layers.',
            'calibration':{'samples':profile['calibration_samples'],'coverage_complete':True,
                'evaluation_checked':True,'evaluation_overlap':0,
                'provenance':f"Original frozen experiment records report {profile['calibration_samples']} samples and no calibration/development overlap; no recalibration during import."},
            **({'training':profile['training']} if 'training' in profile else {})},
        'frozen_candidate':profile,
        'qualification':{'status':'candidate','independent_accuracy_verified':False,
            'selection_scope':'Selected on the existing 2000-image development comparison.'}}


def validate_package(package, release, precision):
    """Accept the reviewed files and exact original-exporter manifest together."""
    package, release = Path(package), Path(release)
    profile = profiles()[precision]
    _verify_model(package, profile)
    manifest = json.loads((package/'manifest.json').read_text())
    if manifest != _manifest(release, profile, precision):
        raise ValueError('Frozen digit manifest differs from the public candidate contract')
    return manifest


def import_model(source, destination, release, precision):
    """Only the exact reviewed graph/weight pair may enter a candidate package."""
    profile = profiles()[precision]
    if source is None:
        raise ValueError(f'{precision} requires --model-source=/path/to/frozen/model. '
                         'The original weights are immutable; no fitting is performed.')
    source, destination, release = Path(source).resolve(), Path(destination), Path(release)
    _verify_model(source, profile)
    manifest = _manifest(release, profile, precision)
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in profile['sha256']:
        shutil.copy2(source/name,destination/name)
        if hashlib.sha256((destination/name).read_bytes()).hexdigest()!=profile['sha256'][name]:
            raise ValueError('Frozen source changed during import: '+name)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
