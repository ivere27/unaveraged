#!/usr/bin/env python3
"""Joint Conv/Linear weight QAT through published VolvoxAI; external data only.

Manifests contain items with path, file_sha256, input_sha256, targets (16 IDs),
and source (real or synthetic). Calibration items need no targets. Output remains
local: it includes checkpoint paths and must not be published wholesale.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from safetensors.numpy import load_file
import volvoxai as vx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shared.cpu_policy import resolve_cpu_threads
from shared.model_support import preprocess, sha256, write_json, packaged_library_paths


def decode(logits):
    if logits.shape!=(1,16,11) or not np.isfinite(logits).all():raise ValueError('Invalid digit logits')
    slots=logits.argmax(-1)[0]
    def digits(values):
        end=np.flatnonzero(values==10)
        return ''.join(str(int(v)) for v in values[:int(end[0]) if len(end) else len(values)])
    return {'phone':digits(slots[:12]),'street':digits(slots[12:])},slots


def read_items(path):
    data=json.loads(path.read_text())
    return data['items'] if isinstance(data,dict) else data


def training_graph(original, batch):
    graph=deepcopy(original)
    if batch==1:return graph
    for value in graph['inputs'].values():
        if value['shape'][0]!=1:raise ValueError('Expected batch-one source')
        value['shape'][0]=batch
    dependent=set(graph['inputs'])
    for node in graph['nodes']:
        varies=bool(dependent.intersection(node['inputs'].values()))
        if not varies:
            continue
        for value in node['outputs'].values():
            if value['shape'][0]!=1:raise ValueError('Unexpected source batch dimension')
            value['shape'][0]=batch
            dependent.add(value['tensor'])
        if node['opType']=='Reshape':
            node['params']['shape'][0]=batch
    return graph


def qat_policy(graph, conv_bits, linear_bits):
    return vx.pb.QatOptions(nodes=[vx.pb.QatNode(node_id=node['id'],
        weights=vx.pb.WeightQuantization(format=vx.pb.WeightFormat(
            bits=conv_bits if node['opType']=='Conv2D' else linear_bits,
            group_size=0 if node['opType']=='Conv2D' else 32),
            algorithm=vx.pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP))
        for node in graph['nodes'] if node['opType'] in ('Conv2D','Linear')])


def learned_names(graph):
    names={name for node in graph['nodes'] if node['opType'] in ('Conv2D','Linear','GroupNorm','LayerNorm')
           for role,name in node['inputs'].items() if role in ('weight','bias')}
    # Learned spatial positions and readout queries; division constants stay fixed.
    produced={v['tensor'] for n in graph['nodes'] for v in n['outputs'].values()}
    for node in graph['nodes']:
        if node['opType']=='Add':
            names.update(n for n in node['inputs'].values() if n not in produced and n not in graph['inputs'])
    return sorted(names)


def pixels(item):
    path=Path(item['path'])
    if sha256(path.read_bytes())!=item['file_sha256']:raise ValueError('Input file changed')
    value=preprocess(path)
    if sha256(value.tobytes())!=item['input_sha256']:raise ValueError('Input preprocessing differs from audit')
    return value


def quantize_candidate(masters, destination, calibration, conv_bits, linear_bits, backend, *, cpu_threads=None):
    cpu_threads=resolve_cpu_threads(cpu_threads)
    graph=json.loads((masters/'graph.json').read_text())
    ops={n['opType'] for n in graph['nodes']}
    rules=[vx.pb.QuantizationRule(operator_name=op,keep_float=vx.pb.Empty()) for op in sorted(ops-{'Linear'})]
    linear=destination.parent/(destination.name+'-linear')
    vx.quantize(masters,output=linear,weight_bits=linear_bits,group_size=32,
        algorithm='mse-clip',rules=rules,cpu_threads=cpu_threads)
    graph=json.loads((linear/'graph.json').read_text())
    rules=[vx.pb.QuantizationRule(operator_name=op,keep_float=vx.pb.Empty())
           for op in sorted({n['opType'] for n in graph['nodes']}-{'Conv2D'})]
    result=vx.quantize(linear,output=destination,weight_bits=conv_bits,group_size=0,algorithm='mse-clip',
        rules=rules,static_activations=vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_I8,
            scheme=vx.pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC),
        calibration_data=({'input0':pixels(item)} for item in calibration),
        calibration_backend=backend,cpu_threads=cpu_threads)
    if result.calibration_backend!=backend:raise ValueError('Calibration backend changed')
    shutil.rmtree(linear)
    return {f:sha256((destination/f).read_bytes()) for f in ('graph.json','model.safetensors')}


def evaluate(package, rows, *, cpu_threads=None):
    cpu_threads=resolve_cpu_threads(cpu_threads)
    groups={name:{'samples':0,'record_correct':0,'slots_correct':0,'cross_entropy_sum':0.} for name in ('all','real','synthetic')}
    with vx.InferenceSession(package,backend='cpu',cpu_threads=cpu_threads,packed_arithmetic='dynamic-i8') as model:
        for item in rows:
            logits=model.run({'input0':pixels(item)})['slot_logits']
            record,slots=decode(logits)
            target=np.asarray(item['targets'],np.int32)
            z=logits[0].astype(np.float64);z-=z.max(axis=1,keepdims=True)
            ce=float(np.mean(np.log(np.exp(z).sum(axis=1))-z[np.arange(16),target]))
            # Exact decoded records, including ignored trailing blank slots.
            expected,_=decode(np.eye(11,dtype=np.float32)[target][None])
            correct=record==expected
            for key in ('all','real' if item['source']=='real' else 'synthetic'):
                group=groups[key];group['samples']+=1;group['record_correct']+=correct
                group['slots_correct']+=int((slots==target).sum())
                group['cross_entropy_sum']+=ce
    for group in groups.values():
        group['record_accuracy']=group['record_correct']/group['samples'] if group['samples'] else None
        group['cross_entropy']=group.pop('cross_entropy_sum')/group['samples'] if group['samples'] else None
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('model','train','validation','calibration','out'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--conv-bits',type=int,choices=(2,3,4),required=True)
    p.add_argument('--linear-bits',type=int,choices=(2,3,4),required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--samples-per-epoch',type=int,default=4096)
    p.add_argument('--learning-rate',type=float,default=3e-5)
    p.add_argument('--seed',type=int,default=20261025)
    p.add_argument('--backend',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--calibration-backend',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--cpu-threads', help='Native CPU threads; defaults to JOBS or available CPUs minus two')
    p.add_argument('--allow-calibration-overlap', action='store_true',
                   help='Allow calibration samples to overlap training (validation remains strictly disjoint)')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    try:a.cpu_threads=resolve_cpu_threads(a.cpu_threads)
    except ValueError as error:p.error(str(error))
    if min(a.epochs,a.batch_size,a.samples_per_epoch)<1 or a.learning_rate<=0:p.error('Positive training settings required')
    if a.samples_per_epoch%a.batch_size:p.error('samples-per-epoch must be divisible by the fixed training batch size')
    a.out.mkdir(parents=True,exist_ok=False)
    train,validation,calibration=[read_items(path) for path in (a.train,a.validation,a.calibration)]
    sets=[{r['input_sha256'] for r in rows} for rows in (train,validation,calibration)]
    if sets[0]&sets[1] or sets[1]&sets[2]:raise ValueError('Validation overlaps training or calibration')
    if not a.allow_calibration_overlap and (sets[0]&sets[2]):raise ValueError('Training/calibration overlap')
    graph=json.loads((a.model/'graph.json').read_text());weights=load_file(a.model/'model.safetensors')
    if Counter(n['opType'] for n in graph['nodes'])['Conv2D']!=11:raise ValueError('Expected FP32 digit source')
    names=learned_names(graph)
    source=a.out/'training-model';source.mkdir()
    write_json(source/'graph.json',training_graph(graph,a.batch_size))
    shutil.copy2(a.model/'model.safetensors',source/'model.safetensors')
    report={'format':'unaveraged-joint-qat/v1','status':'running','private':True,
        'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'scope':'Weight-only joint Conv/Linear QAT; activation QAT is unsupported; static A8 fitted after training.',
        'train_counts':dict(Counter(r['source'] for r in train)),'validation_counts':dict(Counter(r['source'] for r in validation)),
        'calibration_samples':len(calibration),'trainable_names':names,
        'source_sha256':{f:sha256((a.model/f).read_bytes()) for f in ('graph.json','model.safetensors')},
        'runtime_sha256':{f.name:sha256(f.read_bytes()) for f in packaged_library_paths()},'epochs':[]}
    rng=np.random.default_rng(a.seed);start=time.perf_counter()
    try:
        if not a.smoke:
            initial=a.out/'initial-package'
            identity=quantize_candidate(a.model,initial,calibration,a.conv_bits,a.linear_bits,a.calibration_backend,
                                        cpu_threads=a.cpu_threads)
            report['baseline']={'sha256':identity,'validation':evaluate(initial,validation,cpu_threads=a.cpu_threads)}
            print('BASELINE',json.dumps(report['baseline']),flush=True)
        with vx.TrainingSession(source,backend=a.backend,cpu_threads=a.cpu_threads,trainable_names=names,
            qat=qat_policy(graph,a.conv_bits,a.linear_bits),loss=vx.CrossEntropyLoss('slot_logits'),
            optimizer=vx.AdamW(lr=a.learning_rate,weight_decay=0,max_gradient_norm=1),rng_seed=a.seed) as trainer:
            if trainer.backend!=a.backend:raise ValueError('Training backend changed')
            if a.smoke:
                x=np.concatenate([pixels(item) for item in train[:a.batch_size]])
                y=np.asarray([item['targets'] for item in train[:a.batch_size]],np.int32)
                for step in range(3):
                    t=time.perf_counter()
                    with trainer.step({'input0':x},y,output_names=['slot_logits'],locate_nonfinite=True) as result:
                        print('SMOKE',step,'backend',result.backend,'loss',result.loss,'seconds',time.perf_counter()-t,flush=True)
                        if result.backend!=a.backend or not np.isfinite(result.loss):raise ValueError('Invalid smoke result')
                trainer.save(a.out/'smoke-masters')
            else:
                real=[i for i,r in enumerate(train) if r['source']=='real']
                synthetic=[i for i,r in enumerate(train) if r['source']!='real']
                if not real and not synthetic:raise ValueError('Training examples required')
                best=None
                for epoch in range(1,a.epochs+1):
                    if real and synthetic:
                        order=np.concatenate([rng.choice(real,a.samples_per_epoch//2,replace=True),
                            rng.choice(synthetic,a.samples_per_epoch-a.samples_per_epoch//2,replace=a.samples_per_epoch>2*len(synthetic))])
                    elif real:
                        order=rng.choice(real,a.samples_per_epoch,replace=a.samples_per_epoch>len(real))
                    else:
                        order=rng.choice(synthetic,a.samples_per_epoch,replace=a.samples_per_epoch>len(synthetic))
                    rng.shuffle(order);loss_sum=0.;seen=0;t=time.perf_counter();last_progress=t
                    for offset in range(0,len(order),a.batch_size):
                        items=[train[int(i)] for i in order[offset:offset+a.batch_size]]
                        x=np.concatenate([pixels(item) for item in items]);y=np.asarray([item['targets'] for item in items],np.int32)
                        with trainer.step({'input0':x},y) as result:
                            if result.backend!=a.backend or not result.update_applied or not np.isfinite(result.loss):raise ValueError('Invalid training update')
                            loss_sum+=result.loss*len(items);seen+=len(items)
                        now=time.perf_counter()
                        if seen%256==0 or seen==len(order) or now-last_progress>=30:
                            print('TRAIN',epoch,seen,'/',len(order),'loss',loss_sum/seen,'seconds',now-t,flush=True)
                            last_progress=now
                    ep=a.out/f'epoch-{epoch}';ep.mkdir();masters=ep/'masters';trainer.save(masters)
                    write_json(masters/'graph.json',graph)
                    actual=load_file(masters/'model.safetensors')
                    if any(not np.isfinite(v).all() for v in actual.values()):raise ValueError('Nonfinite exported masters')
                    for name in weights:
                        if name not in names and actual[name].tobytes()!=weights[name].tobytes():raise ValueError('Frozen constant changed')
                    candidate=ep/'package'
                    identity=quantize_candidate(masters,candidate,calibration,a.conv_bits,a.linear_bits,a.calibration_backend,
                                                cpu_threads=a.cpu_threads)
                    metrics=evaluate(candidate,validation,cpu_threads=a.cpu_threads)
                    record={'epoch':epoch,'training_loss':loss_sum/seen,'training_samples':seen,
                        'order_sha256':hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
                        'validation':metrics,'sha256':identity,'graph_weight_bytes':sum((candidate/f).stat().st_size for f in identity),
                        'seconds':time.perf_counter()-t}
                    report['epochs'].append(record)
                    # Real accuracy first when available; fallback to all when training pure synthetic.
                    real_acc=metrics['real']['record_accuracy']
                    real_ce=metrics['real']['cross_entropy']
                    rank=(real_acc if real_acc is not None else metrics['all']['record_accuracy'],
                        metrics['all']['record_accuracy'],
                        -real_ce if real_ce is not None else -metrics['all']['cross_entropy'],
                        -metrics['all']['cross_entropy'])
                    if best is None or rank>best[0]:
                        best=(rank,epoch)
                        if (a.out/'best').exists():shutil.rmtree(a.out/'best')
                        shutil.copytree(candidate,a.out/'best')
                        checkpoint=a.out/f'best-checkpoint-epoch-{epoch}.pb'
                        trainer.save_checkpoint(checkpoint)
                        report['best_epoch']=epoch
                        report['best_checkpoint']=checkpoint.name
                    print('EPOCH',json.dumps(record),flush=True)
                    shutil.rmtree(masters)
                    write_json(a.out/'report.json',report)
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error),
            'node_index':getattr(error,'node',None),'report':str(getattr(error,'report',''))})
        print('ERROR_DETAIL',report['error'],flush=True)
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        write_json(a.out/'report.json',report)


if __name__=='__main__':main()
