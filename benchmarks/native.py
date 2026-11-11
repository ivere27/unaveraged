#!/usr/bin/env python3
"""Compare F32, W8A8 and frozen candidates on identical accuracy/timing inputs."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import volvoxai as vx
from shared.model_support import packaged_library_paths, preprocess, sha256, write_json
from digit.check import decode, load_cases
from benchmarks.models import model_definitions


def paired_counts(pairs):
    losses = sum(a and not b for a,b in pairs)
    gains = sum(b and not a for a,b in pairs)
    n = len(pairs)
    delta = (losses-gains)/n
    se = math.sqrt(max(0, (losses+gains)/n-delta*delta)/n)
    return {'samples':n, 'new_errors':losses, 'corrected_errors':gains,
        'net_additional_errors':losses-gains, 'observed_loss_percentage_points':100*delta,
        'approximate_one_sided_95_upper_loss_percentage_points':100*(delta+1.644854*se),
        'interval_method':'Paired normal approximation; descriptive only, does not adjust for candidate selection or label error.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-root', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--cycles', type=int, default=15)
    parser.add_argument('--warmups', type=int, default=4)
    parser.add_argument('--out', type=Path, default=ROOT/'.cache/benchmarks/retest')
    args = parser.parse_args()
    if args.cycles<1 or args.warmups<1 or args.limit is not None and args.limit<1:
        parser.error('Cycles, warmups and limit must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    definitions, packages = model_definitions()
    config = json.loads((ROOT/'digit/extension.json').read_text())
    source = ROOT/'.cache/digit/huggingface'/config['model']['revision']
    spec = importlib.util.spec_from_file_location('benchmark_question_router',source/'question_router.py')
    router = importlib.util.module_from_spec(spec); spec.loader.exec_module(router)
    cases = load_cases(source,router,args.evaluation_root,args.limit)
    report = {'format':'unaveraged-native-benchmark/v1','created_at':datetime.now(timezone.utc).isoformat(),
        'status':'running','runtime':{'package':'volvoxai','version':vx.__version__,
            'sha256':{p.name:sha256(p.read_bytes()) for p in packaged_library_paths()}},
        'environment':{'python':platform.python_version(),'platform':platform.platform(),
            'machine':platform.machine(),'cpu':next((line.split(':',1)[1].strip() for line in
                Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown')},
        'evaluation':{'kind':'existing-development-retest' if args.evaluation_root else 'public-examples',
            'samples':len(cases),'independent':False,'primary_metric':'Exact full phone+street record',
            'label_caveat':'Provided annotations; partial prior review, not a fully human-audited gold dataset.',
            'used_for_previous_selection':bool(args.evaluation_root)},
        'models':{},'latency':{'backend':'cpu','cpu_threads':1,'cycles':args.cycles,
            'warmups_per_input':args.warmups,'inputs':3,'input_source':'Two public HF receipts and a deterministic synthetic gradient',
            'timer_includes':['InferenceSession.run, input/output transport and owned output arrays'],
            'timer_excludes':['preprocessing','load','compile','context creation','correctness validation'],
            'order':'Alternate forward/reverse model order each cycle','host_idle_claimed':False},
        'selection_scope':'Frozen models; no refitting or candidate selection during this retest.'}
    private=[]; states={}; input_digest=hashlib.sha256(); label_digest=hashlib.sha256()
    audit=json.loads((packages['w8a8']/'calibration-inputs.json').read_text())
    calibration_hashes={item['input_sha256'] for item in audit['images']}
    try:
        with ExitStack() as stack:
            for name,profile in definitions.items():
                directory=packages[name]
                for file,expected in profile['sha256'].items():
                    if sha256((directory/file).read_bytes())!=expected:raise ValueError('Frozen model hash mismatch: '+name)
                arithmetic=profile['packed_arithmetic']
                expected=(vx.pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8 if arithmetic=='dynamic-i8'
                          else vx.pb.PackedArithmetic.PACKED_ARITHMETIC_F32)
                session=stack.enter_context(vx.InferenceSession(directory,backend='cpu',cpu_threads=1,packed_arithmetic=arithmetic))
                if session.packed_arithmetic!=expected:
                    raise ValueError('Compiled arithmetic differs from model profile: '+name)
                states[name]=session
                report['models'][name]={'sha256':profile['sha256'],'graph_weight_bytes':profile['graph_weight_bytes'],
                    'packed_arithmetic':arithmetic,'conv_weight_bits':profile['conv_weight_bits'],
                    'linear_weight_bits':profile['linear_weight_bits'],
                    'calibration_samples':profile['calibration_samples'],
                    'calibration_evaluation_overlap':0 if name=='w8a8' else None,
                    'source_preparation':profile.get('source_preparation','Frozen low-bit experiment'),
                    'record_correct':0,'phone_correct':0,'street_correct':0,
                    'answer_correct':0,'answer_total':0}
            for index,case in enumerate(cases):
                values=preprocess(case['image']);input_hash=sha256(values.tobytes());input_digest.update(bytes.fromhex(input_hash))
                report['models']['w8a8']['calibration_evaluation_overlap']+=input_hash in calibration_hashes
                label_digest.update(json.dumps(case['record'],sort_keys=True).encode())
                row={'id':case['id'],'truth':case['record'],'models':{}}
                for name,session in states.items():
                    record,slots=decode(session.run({'input0':values})['slot_logits'])
                    good=record==case['record'];m=report['models'][name];m['record_correct']+=good
                    for field in ('phone','street'):m[field+'_correct']+=record[field]==case['record'][field]
                    for question in case['questions']:
                        answer=router.answer_from_record(question['question'],**record)
                        m['answer_correct']+=str(answer)==str(question['answer']);m['answer_total']+=1
                    row['models'][name]={'record_correct':good,'record':record}
                private.append(row)
                if (index+1)%100==0 or index+1==len(cases):print('EVALUATED',index+1,'/',len(cases),flush=True)
            report['evaluation'].update(ordered_inputs_sha256=input_digest.hexdigest(),ordered_labels_sha256=label_digest.hexdigest())
            for name,m in report['models'].items():
                m.update(record_accuracy=m['record_correct']/len(cases),phone_accuracy=m['phone_correct']/len(cases),
                    street_accuracy=m['street_correct']/len(cases),answer_accuracy=m['answer_correct']/m['answer_total'])
                for reference,key in (('f32','f32'),('w8a8','w8a8')):
                    m['paired_vs_'+key]=paired_counts([(r['models'][reference]['record_correct'],r['models'][name]['record_correct']) for r in private])
            if report['models']['w8a8']['calibration_evaluation_overlap']:
                raise ValueError('W8A8 calibration overlaps evaluated inputs')
            public=json.loads((source/'examples/examples.json').read_text())
            inputs=[preprocess(source/'examples'/p['image']) for p in public]
            gradient=np.linspace(-1,1,320*672,dtype=np.float32).reshape(1,1,320,672)
            inputs=(inputs+[gradient])[:3]
            wasm_spec={'format':'unaveraged-wasm-fixtures/v1','cycles':args.cycles,'warmups':args.warmups,
                'rtol':.0005,'atol':.001,'models':{},'cases':[]}
            for i,x in enumerate(inputs):
                (args.out/f'input-{i}.f32').write_bytes(x.tobytes())
                wasm_spec['cases'].append({'file':f'input-{i}.f32','sha256':sha256(x.tobytes()),'references':{}})
            for name,session in states.items():
                wasm_spec['models'][name]={'directory':str(packages[name].resolve()),**definitions[name]}
                for i,x in enumerate(inputs):
                    actual=session.run({'input0':x})['slot_logits'];raw=actual.tobytes()
                    wasm_spec['cases'][i]['references'][name]={'values':actual.reshape(-1).tolist(),
                        'slots':actual.argmax(-1).reshape(-1).tolist(),'sha256':sha256(raw)}
                    for _ in range(args.warmups):
                        if session.run({'input0':x})['slot_logits'].tobytes()!=raw:raise ValueError('Native output changed on warmup')
            samples={name:[[] for _ in inputs] for name in states}
            names=list(states)
            for cycle in range(args.cycles):
                for i,x in enumerate(inputs):
                    for name in (names if cycle%2==0 else list(reversed(names))):
                        started=time.perf_counter_ns();output=states[name].run({'input0':x})['slot_logits']
                        elapsed=(time.perf_counter_ns()-started)/1e6;samples[name][i].append(elapsed)
                        if sha256(output.tobytes())!=wasm_spec['cases'][i]['references'][name]['sha256']:
                            raise ValueError('Native measured output changed')
            for name,m in report['models'].items():
                m['latency']={'samples_ms_per_input':samples[name],
                    'median_ms_per_input':[statistics.median(a) for a in samples[name]],
                    'pooled_median_ms':statistics.median(sum(samples[name],[]))}
            write_json(args.out/'fixtures.json',wasm_spec)
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        # The aggregate report intentionally contains no paths, receipt IDs, labels or predictions.
        write_json(args.out/'native.json',report)
        write_json(args.out/'native-private.json',{'private':True,'do_not_publish':True,'cases':private})
    print('NATIVE COMPLETE', {name:m['record_correct'] for name,m in report['models'].items()},flush=True)


if __name__=='__main__':main()
