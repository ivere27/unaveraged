"""Candidate identity and API contract regressions."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from digit.frozen import profiles, validate_graph, import_model


class FrozenTests(unittest.TestCase):
    def test_default_qat_identity_matches_selected_joint_training_checkpoint(self):
        profile = profiles()['conv4-linear2']
        report = json.loads((ROOT/'benchmarks/results.json').read_text())
        identity = report['models']['digit']['identity']
        self.assertEqual(profile['sha256'], identity['sha256'])
        self.assertEqual(profile['method'], 'volvoxai-qat')
        self.assertEqual(profile['training']['selected_epoch'], identity['selected_epoch'])

    def test_default_builds_resolve_independent_model_profiles(self):
        import argparse
        from shared.build import selected_builds
        args = argparse.Namespace(model='all', precision='default', variant='ptq',
            model_source=None, package_only=True, fixtures=False)
        builds = selected_builds(args)
        self.assertEqual([(c['id'], a.precision) for c, a in builds],
                         [('digit', 'conv4-linear2'), ('vqa', 'w4a8')])
        args.variant = 'fp32'
        self.assertEqual(len(selected_builds(args)), 2)

    def test_frozen_selection_has_distinct_immutable_identities(self):
        data=profiles()
        self.assertEqual(set(data),{'conv8-linear4','conv4-linear4','conv4-linear3','conv4-linear2'})
        self.assertEqual(len({v['sha256']['model.safetensors'] for v in data.values()}),4)
        for profile in data.values():
            self.assertEqual(profile['packed_arithmetic'],'dynamic-i8')
            self.assertEqual(profile['linear_group_size'],32)
            self.assertEqual(set(profile['sha256']),{'graph.json','model.safetensors'})

    def test_import_refuses_corruption_before_publishing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';source.mkdir()
            (source/'graph.json').write_text('{}');(source/'model.safetensors').write_bytes(b'wrong model')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                import_model(source,root/'output',root/'release','conv4-linear3')
            self.assertFalse((root/'output').exists())
            with self.assertRaisesRegex(ValueError,'requires --model-source'):
                import_model(None,root/'output',root/'release','conv4-linear4')

    def test_graph_precision_is_checked_separately_from_metadata(self):
        profile=profiles()['conv4-linear3']
        graph={'inputs':{'input0':{'shape':[1,1,320,672],'dtype':'float32'}},'outputs':['slot_logits'],
            'nodes':[{'opType':'QConv2D','params':{'weight_bits':4}} for _ in range(11)]+
                    [{'opType':'PackedLinear','params':{'weight_bits':3,'group_size':32}} for _ in range(7)]}
        validate_graph(graph,profile)
        graph['nodes'][-1]['params']['weight_bits']=4
        with self.assertRaisesRegex(ValueError,'linear precision'):
            validate_graph(graph,profile)

    def test_f32_and_w8a8_baselines_cannot_hide_packed_dense_layers(self):
        from benchmarks.models import validate_baseline_graph
        for name,operator in (('f32','Conv2D'),('w8a8','QConv2D')):
            graph={'inputs':{'input0':{'shape':[1,1,320,672],'dtype':'float32'}},'outputs':['slot_logits'],
                'nodes':[{'opType':operator,'params':{'weight_bits':8}} for _ in range(11)]+
                        [{'opType':'Linear','params':{}} for _ in range(7)]}
            validate_baseline_graph(graph,name)
            graph['nodes'][-1]['opType']='PackedLinear'
            with self.assertRaisesRegex(ValueError,'seven FP32 Linear'):
                validate_baseline_graph(graph,name)
        graph['nodes'][-1]['opType']='Linear'
        with self.assertRaisesRegex(ValueError,'quantized operators'):
            validate_baseline_graph(graph,'f32')

    def test_paired_error_counts_use_same_records(self):
        spec=importlib.util.spec_from_file_location('bench_native',ROOT/'benchmarks/native.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        result=module.paired_counts([(True,False),(True,False),(False,True),(True,True)])
        self.assertEqual(result['new_errors'],2)
        self.assertEqual(result['corrected_errors'],1)
        self.assertEqual(result['observed_loss_percentage_points'],25)

    def test_frozen_profile_cannot_be_applied_to_vqa_or_fp32_variant(self):
        for extra in (['--model','vqa'],['--model','digit','--variant','fp32']):
            result=subprocess.run([sys.executable,str(ROOT/'shared/build.py'),'--precision','conv4-linear4',
                                   '--check-inputs',*extra],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('Frozen candidate profiles require',result.stderr)

if __name__=='__main__':unittest.main()
