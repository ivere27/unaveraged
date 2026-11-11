"""Heldout checks use the extension's answer contract and complete input sets."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vqa import check


VALID_OUTPUT = '<field>phone</field><op>front_1</op><answer>7</answer>'


class HeldoutVqaTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'images').mkdir()
        (self.root / 'annotations').mkdir()
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2]
        self.tokenizer.decode.return_value = VALID_OUTPUT
        self.prediction = {'tokens': [1, 10, 2], 'family': 'phone'}
        self.addCleanup(patch.stopall)
        patch.object(check, '_worker_tok', self.tokenizer, create=True).start()
        patch.object(check, '_worker_runner', None, create=True).start()
        self.preprocess = patch.object(check, 'preprocess', return_value=None).start()
        patch.object(check, 'generate', return_value=self.prediction).start()
        self.pool = patch('multiprocessing.Pool').start()
        self.pool.return_value.__enter__.return_value.imap.side_effect = (
            lambda function, items, chunksize: map(function, items))
        self.output = io.StringIO()
        patch('sys.stdout', self.output).start()

    def annotation(self, name='record', *, suffix='.jpg', image_path=None, answer='7'):
        data = {'question': 'What is the first phone number digit?', 'answer': answer}
        if image_path is not None:
            data['image_path'] = image_path
        (self.root / 'annotations' / f'{name}.json').write_text(json.dumps(data))
        image = self.root / (image_path or f'images/{name}{suffix}')
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b'image fixture; preprocessing is mocked')
        return image

    def evaluate(self, **kwargs):
        return check.run_heldout_evaluation(self.root / 'packages', 'w4a8', self.root, **kwargs)

    def test_accepts_only_complete_eos_terminated_structured_answers(self):
        cases = [
            (VALID_OUTPUT, [1, 10, 2], True),
            ('<field>phone</field><value>7123</value><op>front_1</op><answer> 7 </answer>', [1, 10, 2], True),
            ('<answer>7</answer>', [1, 10, 2], False),
            (VALID_OUTPUT + 'extra', [1, 10, 2], False),
            ('prefix' + VALID_OUTPUT, [1, 10, 2], False),
            (VALID_OUTPUT, [1, 10], False),
            (VALID_OUTPUT, [], False),
            (VALID_OUTPUT.replace('>7<', '>8<'), [1, 10, 2], False),
        ]
        for text, tokens, correct in cases:
            with self.subTest(text=text, tokens=tokens):
                self.tokenizer.decode.return_value = text
                self.prediction['tokens'] = tokens
                result = check._eval_record(('fixture.jpg', 'phone question', '7', 'record'))
                self.assertEqual(result['correct'], correct)
                self.assertEqual(result['finished'], bool(tokens and tokens[-1] == 2))

    def test_invalid_utf8_is_incorrect_instead_of_replacement_decoded(self):
        self.tokenizer.decode.side_effect = lambda tokens, *, errors: b'\xff'.decode('utf-8', errors=errors)
        result = check._eval_record(('fixture.jpg', 'phone question', '7', 'record'))
        self.assertFalse(result['correct'])
        self.assertFalse(result['structured_valid'])
        self.tokenizer.decode.assert_called_once_with(self.prediction['tokens'], errors='strict')

    def test_rejects_empty_dataset_before_starting_workers(self):
        with self.assertRaisesRegex(ValueError, 'no annotation records'):
            self.evaluate()
        self.pool.assert_not_called()

    def test_rejects_missing_images_instead_of_shrinking_denominator(self):
        self.annotation('first')
        self.annotation('second').unlink()
        with self.assertRaisesRegex(ValueError, 'matching image: second.json'):
            self.evaluate()
        self.pool.assert_not_called()

    def test_explicit_missing_image_is_not_replaced_with_a_different_image(self):
        self.annotation(image_path='images/selected.png').unlink()
        (self.root / 'images' / 'record.jpg').write_bytes(b'wrong image')
        with self.assertRaisesRegex(ValueError, 'image is missing'):
            self.evaluate()
        self.pool.assert_not_called()

    def test_rejects_missing_labels_before_starting_workers(self):
        self.annotation(answer=None)
        with self.assertRaisesRegex(ValueError, 'requires question and answer'):
            self.evaluate()
        self.pool.assert_not_called()

    def test_rejects_nonpositive_limits_and_worker_counts(self):
        self.annotation()
        for option in ('limit', 'workers'):
            for value in (0, -1):
                with self.subTest(option=option, value=value):
                    with self.assertRaisesRegex(ValueError, f'--{option} must be positive'):
                        self.evaluate(**{option: value})
        self.pool.assert_not_called()

    def test_uses_explicit_png_path_and_limits_selected_annotations(self):
        image = self.annotation('first', image_path='images/different-name.png')
        self.annotation('second').unlink()
        report = self.evaluate(limit=1)
        self.preprocess.assert_called_once_with(image)
        self.assertEqual((report['correct'], report['total']), (1, 1))

    def test_resolves_png_and_records_metrics_without_claiming_accuracy_pass(self):
        image = self.annotation(suffix='.png', answer='8')
        report_dir = self.root / 'reports'
        report = self.evaluate(report_dir=report_dir)
        self.preprocess.assert_called_once_with(image)
        self.assertEqual((report['correct'], report['total'], report['accuracy']), (0, 1, 0.0))
        self.assertEqual(report['status'], 'completed')
        self.assertNotIn('PASS', self.output.getvalue())
        self.assertEqual(json.loads((report_dir / 'w4a8' / 'heldout-check.json').read_text()), report)


if __name__ == '__main__':
    unittest.main()
