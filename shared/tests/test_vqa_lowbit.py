"""Recovery must preserve completed VQA cases and reject mixed execution runs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.vqa_lowbit import read_checkpoint, verify_resume, write_report


class VQAEvaluationRecoveryTests(unittest.TestCase):
    def test_interrupted_last_row_is_replayed_without_losing_completed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.jsonl'
            complete = b'{"id":"one","answer_correct":true}\n'
            path.write_bytes(complete + b'{"id":"two","answer_cor')
            self.assertEqual(read_checkpoint(path), [{'id': 'one', 'answer_correct': True}])
            self.assertEqual(path.read_bytes(), complete)
            with path.open('ab') as stream:
                stream.write(b'{"id":"two","answer_correct":false}\n')
            self.assertEqual(len(read_checkpoint(path)), 2)

    def test_corruption_of_committed_row_is_never_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.jsonl'
            original = b'{"id":"one"}\nnot-json\n'
            path.write_bytes(original)
            with self.assertRaises(json.JSONDecodeError):
                read_checkpoint(path)
            self.assertEqual(path.read_bytes(), original)

    def test_resume_rejects_backend_thread_and_tokenizer_changes(self):
        report = {'model': {'hash': 'model'}, 'ordered_cases_sha256': 'cases',
                  'runtime_sha256': 'runtime', 'tokenizer_sha256': 'tokenizer',
                  'backend': 'cpu', 'cpu_threads': 1, 'batch_size': 1,
                  'max_decoder_steps': 191, 'profile': 'w4', 'split': 'validation',
                  'shard_index': 0, 'shards': 1}
        verify_resume(report, report.copy())
        for key, value in [('backend', 'cuda'), ('cpu_threads', 4), ('tokenizer_sha256', 'different')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                verify_resume(report, {**report, key: value})

    def test_atomic_report_replacement_leaves_only_complete_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            write_report(path, {'status': 'running'})
            write_report(path, {'status': 'completed', 'samples': 2})
            self.assertEqual(json.loads(path.read_text()), {'status': 'completed', 'samples': 2})
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['report.json'])


if __name__ == '__main__':
    unittest.main()
