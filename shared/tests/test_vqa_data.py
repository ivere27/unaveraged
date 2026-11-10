"""VQA evaluation must reject malformed answers and prevent image leakage."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.vqa_data import (assert_disjoint, load_records, prepare_calibration,
    score_prediction, select_records, split_train_validation, structured_target)


def record(index, family='phone', image=None, input_hash=None):
    return {'id': str(index), 'receipt_id': str(index), 'image': image or Path(f'/fixture/{index}.jpg'),
            'question': f'Question {index}', 'answer': '7', 'family': family, 'language': 'en',
            'field': 'phone', 'op': 'front_1',
            'target': '<field>phone</field><value>7123</value><op>front_1</op><answer>7</answer>',
            'file_sha256': f'file-{index}', 'input_sha256': input_hash or f'input-{index}'}


class VQADataTests(unittest.TestCase):
    def test_loader_preserves_label_and_support_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'annotations').mkdir(); (root / 'images').mkdir()
            annotations = [
                {'question': '전화번호 앞에서 1번째 숫자는?', 'answer': '7',
                 'receipt': {'store': {'phone': '712-3'}}},
                {'question': 'What is the store name?', 'answer': 'Fixture'},
            ]
            for index, annotation in enumerate(annotations):
                (root / 'annotations' / f'{index}.json').write_text(json.dumps(annotation))
                (root / 'images' / f'{index}.jpg').write_bytes(b'fixture')
            rows, summary = load_records(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['answer'], '7')
            self.assertEqual(rows[0]['target'], record(0)['target'])
            self.assertEqual(summary['skipped'], {'unsupported_question': 1})

    def test_current_target_schema_omits_redundant_direct_read_value(self):
        self.assertEqual(structured_target('store', 'Fixture', 'identity', 'Fixture'),
                         '<field>store</field><op>identity</op><answer>Fixture</answer>')
        self.assertIn('<value>7123</value>', structured_target('phone', '7123', 'front_1', '7'))

    def test_product_metric_rejects_unfinished_and_partial_structures(self):
        row = record(0)
        self.assertTrue(score_prediction(row, row['target'])['answer_correct'])
        self.assertFalse(score_prediction(row, row['target'], finished=False)['answer_correct'])
        self.assertFalse(score_prediction(row, '<answer>7</answer>')['answer_correct'])
        self.assertFalse(score_prediction(row, row['target'] + 'extra')['answer_correct'])

    def test_sampling_is_stratified_deterministic_and_order_independent(self):
        rows = [record(i, 'phone' if i < 8 else 'address') for i in range(16)]
        selected = select_records(rows, 4, 31)
        self.assertEqual(selected, select_records(list(reversed(rows)), 4, 31))
        self.assertEqual(sum(r['family'] == 'phone' for r in selected), 2)
        self.assertEqual(len({r['id'] for r in selected}), 4)

    def test_split_keeps_duplicate_inputs_together(self):
        rows = [record(i) for i in range(12)] + [record(99, input_hash='input-3')]
        train, validation, audit = split_train_validation(rows, .25, 17)
        self.assertFalse({r['input_sha256'] for r in train} & {r['input_sha256'] for r in validation})
        self.assertEqual(audit['validation_unique_inputs'], 3)
        self.assertEqual(len(train) + len(validation), len(rows))

    def test_calibration_excludes_different_files_with_identical_model_inputs(self):
        train = [record(i) for i in range(8)] + [record(20, input_hash='input-2')]
        evaluation = [record(100, input_hash='input-1')]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            assert_disjoint(train, evaluation)
        selected, audit = prepare_calibration(train, evaluation, 6, 12)
        self.assertNotIn('input-1', {r['input_sha256'] for r in selected})
        self.assertEqual(len({r['input_sha256'] for r in selected}), 6)
        self.assertEqual(audit['excluded_overlapping_records'], 1)
        self.assertEqual(audit['deduplicated_training_records'], 1)
        with self.assertRaisesRegex(ValueError, 'Only 7'):
            prepare_calibration(train, evaluation, 8)


if __name__ == '__main__':
    unittest.main()
