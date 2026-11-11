"""Local model preparation respects available CPUs and explicit user budgets."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from shared.cpu_policy import available_cpu_count, resolve_cpu_threads


class CpuPolicyTests(unittest.TestCase):
    def test_auto_respects_affinity_and_reserves_two_cpus(self):
        with patch('shared.cpu_policy.os.sched_getaffinity', return_value=set(range(12)), create=True), \
             patch('shared.cpu_policy.os.cpu_count', return_value=128):
            self.assertEqual(available_cpu_count(), 12)
            self.assertEqual(resolve_cpu_threads('auto'), 10)
        for count in (1, 2, 3):
            with self.subTest(count=count), patch('shared.cpu_policy.available_cpu_count', return_value=count):
                self.assertEqual(resolve_cpu_threads('auto'), 1)

    def test_platform_fallback_and_explicit_override(self):
        with patch('shared.cpu_policy.os.sched_getaffinity', side_effect=OSError, create=True), \
             patch('shared.cpu_policy.os.process_cpu_count', return_value=8, create=True):
            self.assertEqual(resolve_cpu_threads('auto'), 6)
        with patch('shared.cpu_policy.available_cpu_count', return_value=2), \
             patch.dict(os.environ, {'JOBS': '4'}):
            self.assertEqual(resolve_cpu_threads(), 4)
            self.assertEqual(resolve_cpu_threads('1'), 1)
            self.assertEqual(resolve_cpu_threads(3), 3)

    def test_invalid_budget_fails_before_loading_native_build_dependencies(self):
        for value in (0, -1, True, 1.5, '', '0', '-2', '1.5', 'many'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                resolve_cpu_threads(value)
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, '-I', str(root/'shared/build.py'),
                                 '--check-inputs', '--cpu-threads', '0'],
                                cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('positive integer', result.stderr)


if __name__ == '__main__':
    unittest.main()
