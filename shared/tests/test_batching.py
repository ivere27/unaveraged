"""Batched model rewrites must preserve constant broadcast axes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file
import volvoxai as vx

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from digit.qat import training_graph


class BatchGraphTests(unittest.TestCase):
    def test_constant_query_broadcast_matches_independent_matrix_reference(self):
        graph={'format':'volvox-graph/v1','dimensions':{},
            'inputs':{'x':{'dtype':'float32','shape':[1,2,3]}},'outputs':['logits'],
            'execution':{'stateless':{},'entrypoints':[{'name':'evaluate','effect':1,'allowed_ranges':[1],
                'inputs':['x'],'outputs':['logits']}]},
            'nodes':[
                {'id':'query-transpose','opType':'Transpose','inputs':{'input':'query'},
                 'outputs':{'out':{'tensor':'qt','dtype':'float32','shape':[1,3,2]}},'params':{'perm':[0,2,1]}},
                {'id':'reshape','opType':'Reshape','inputs':{'input':'x'},
                 'outputs':{'out':{'tensor':'features','dtype':'float32','shape':[1,2,3]}},'params':{'shape':[1,2,3]}},
                {'id':'attention','opType':'BatchMatMul','inputs':{'a':'features','b':'qt'},
                 'outputs':{'out':{'tensor':'logits','dtype':'float32','shape':[1,2,2]}},'params':{}}]}
        query=np.array([[[.1,.2,.3],[.4,-.2,.1]]],np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            package=Path(temporary)
            save_file({'query':query},package/'model.safetensors')
            for batch in (1,2,4):
                changed=training_graph(graph,batch)
                self.assertEqual(changed['nodes'][0]['outputs']['out']['shape'],[1,3,2])
                (package/'graph.json').write_text(json.dumps(changed))
                x=(np.arange(batch*6,dtype=np.float32).reshape(batch,2,3)-5)/7
                with vx.InferenceSession(package,backend='cpu',cpu_threads=1) as model:
                    actual=model.run({'x':x})['logits']
                np.testing.assert_allclose(actual,x@query.transpose(0,2,1),rtol=1e-6,atol=1e-6)


if __name__=='__main__':unittest.main()
