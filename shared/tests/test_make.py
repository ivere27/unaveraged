"""The small Make workflow must reuse models for tests/demo and honor selection."""
import os
from pathlib import Path
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]


class MakeCheckTests(unittest.TestCase):
    def dry_run(self, *arguments):
        env = {key: value for key, value in os.environ.items()
               if key not in ('MAKEFLAGS', 'MFLAGS', 'MAKELEVEL')}
        return subprocess.run(
            ['make', '--no-print-directory', '--dry-run', '--assume-old=deps',
             '--assume-old=check-inputs', '--assume-old=.venv/.docs-deps', *arguments],
            cwd=ROOT, env=env, check=True, capture_output=True, text=True).stdout

    def test_default_make_builds_once_and_test_reuses_packages(self):
        default = self.dry_run()
        invocations = [line for line in default.splitlines() if 'shared/build.py ' in line]
        self.assertEqual(len(invocations), 1)
        self.assertIn('--model "all"', invocations[0])
        self.assertIn('--browser "all"', invocations[0])
        for target in ('test', 'check'):
            commands = self.dry_run(target)
            self.assertNotIn('shared/build.py', commands, 'Tests must not train/rebuild model packages')
            self.assertIn('node shared/tests/browser.mjs', commands)
            self.assertIn('node shared/tests/firefox-browser.mjs', commands)
        self.assertEqual(self.dry_run('test'), self.dry_run('check'), 'The old check name remains compatible')

    def test_demo_stages_existing_models_and_serves_only_localhost(self):
        commands = self.dry_run('demo', 'PORT=8123')
        self.assertLess(commands.index('shared/pages.py'), commands.index('http.server'))
        self.assertIn('-m http.server "8123" --bind 127.0.0.1 --directory dist/pages', commands)
        self.assertNotIn('shared/build.py', commands)
        self.assertNotIn('pip install', commands)
        self.assertNotIn('npm', commands)

    def test_help_lists_only_the_small_common_workflow(self):
        result = subprocess.run(['make', '--no-print-directory', 'help'], cwd=ROOT,
                                check=True, capture_output=True, text=True)
        common = [line for line in result.stdout.splitlines() if line.startswith('make ')]
        self.assertEqual(len(common), 7)
        self.assertNotIn('make benchmark', result.stdout)
        self.assertNotIn('make check-firefox', result.stdout)
        self.assertIn('MODEL=all|digit|vqa', result.stdout)
        self.assertIn('BROWSER=all|chrome|firefox', result.stdout)
        self.assertIn('JOBS=auto', result.stdout)

    def test_native_cpu_budget_reaches_build_environment(self):
        env = {key: value for key, value in os.environ.items()
               if key not in ('MAKEFLAGS', 'MFLAGS', 'MAKELEVEL', 'JOBS')}
        probe = 'probe:;@python3 -c \'import os; print(os.environ["JOBS"])\''
        command = ['make', '--no-print-directory', '--eval', probe, 'probe']
        for arguments, expected in (([], 'auto'), (['JOBS=10'], '10'), (['JOBS=1'], '1')):
            with self.subTest(arguments=arguments):
                result = subprocess.run([*command, *arguments], cwd=ROOT, env=env,
                                        check=True, capture_output=True, text=True)
                self.assertEqual(result.stdout.strip(), expected)

    def test_one_release_command_selects_check_create_or_upload(self):
        for action in ('check', 'create', 'upload'):
            with self.subTest(action=action):
                commands = self.dry_run('release', f'ACTION={action}')
                self.assertIn(f'.venv/bin/python shared/release.py "{action}"', commands)
                self.assertNotIn('shared/build.py', commands)

    def test_release_defaults_to_local_model_upload_without_a_tag(self):
        default = self.dry_run('release')
        self.assertIn('shared/release.py "models"', default)
        self.assertNotIn('shared/build.py', default)
        versioned = self.dry_run('release', 'TAG=v1.2.3')
        self.assertIn('shared/release.py "create"', versioned)
        self.assertNotIn('shared/build.py', versioned)

    def test_advanced_build_options_keep_python_defaults_and_explicit_overrides(self):
        # Exporting undefined Make variables would pass empty strings and disable
        # the Python builder's synth/calibration defaults.
        env = {key: value for key, value in os.environ.items()
               if key not in ('MAKEFLAGS', 'MFLAGS', 'MAKELEVEL', 'CALIBRATION_IMAGES', 'CALIBRATION_COUNT')}
        probe = 'probe:;@python3 -c \'import os; print(repr(os.environ.get("CALIBRATION_IMAGES")), repr(os.environ.get("CALIBRATION_COUNT")))\''
        command = ['make', '--no-print-directory', '--eval', probe, 'probe']
        default = subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        self.assertEqual(default.stdout.strip(), 'None None')
        explicit = subprocess.run([*command, 'CALIBRATION_COUNT=128'], cwd=ROOT, env=env,
                                  check=True, capture_output=True, text=True)
        self.assertEqual(explicit.stdout.strip(), "None '128'")

    def test_check_runs_only_selected_browser_runners(self):
        for browser in ('all', 'chrome', 'firefox'):
            with self.subTest(browser=browser):
                result = subprocess.run(
                    ['make', '--no-print-directory', '--dry-run', '--assume-old=deps',
                     'check', f'BROWSER={browser}'],
                    cwd=ROOT, check=True, capture_output=True, text=True)
                self.assertEqual('node shared/tests/browser.mjs ' in result.stdout,
                                 browser != 'firefox')
                self.assertEqual('node shared/tests/firefox-browser.mjs ' in result.stdout,
                                 browser != 'chrome')

    def test_browser_selection_reaches_packaging_without_extra_model_builds(self):
        for target in ('build', 'package', 'dev'):
            for browser in ('all', 'chrome', 'firefox'):
                with self.subTest(target=target, browser=browser):
                    result = subprocess.run(
                        ['make', '--no-print-directory', '--dry-run', '--assume-old=deps',
                         '--assume-old=check-inputs', target, f'BROWSER={browser}'],
                        cwd=ROOT, check=True, capture_output=True, text=True)
                    invocations = [shlex.split(line) for line in result.stdout.splitlines()
                                   if line.startswith('.venv/bin/python shared/build.py ')]
                    self.assertEqual(len(invocations), 1)
                    args = invocations[0]
                    self.assertEqual(args[args.index('--browser') + 1], browser)

    def check_commands(self, model='all', precision='default'):
        env = {key: value for key, value in os.environ.items()
               if key not in ('MAKEFLAGS', 'MFLAGS', 'MAKELEVEL')}
        result = subprocess.run(
            ['make', '--no-print-directory', '--dry-run', '--assume-old=deps',
             'check', f'MODEL={model}', f'PRECISION={precision}'],
            cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        commands = {}
        for line in result.stdout.splitlines():
            for project in ('digit', 'vqa'):
                invocation = f'.venv/bin/python {project}/check.py'
                if invocation in line:
                    commands[project] = shlex.split(line.split(invocation, 1)[1].split(';', 1)[0])
            if line.startswith('node shared/tests/browser.mjs '):
                arguments = shlex.split(line)[2:]
                mode = 'production' if '--production' in arguments else 'lifecycle'
                commands[mode] = arguments
        self.assertEqual(set(commands), {'digit', 'vqa', 'production', 'lifecycle'})
        return commands

    def test_default_checks_use_each_model_default_and_browser_default(self):
        commands = self.check_commands()
        for project in ('digit', 'vqa'):
            self.assertNotIn('--precision', commands[project])
        for mode in ('production', 'lifecycle'):
            arguments = commands[mode]
            self.assertEqual(arguments[arguments.index('--precision') + 1], 'default')

    def test_explicit_precisions_reach_native_and_browser_checks(self):
        for model, precision in (('all', 'w8a8'), ('all', 'all'),
                                 ('digit', 'conv4-linear2'), ('vqa', 'w4a8')):
            with self.subTest(model=model, precision=precision):
                commands = self.check_commands(model, precision)
                for name in (('digit', 'vqa') if model == 'all' else (model,)):
                    arguments = commands[name]
                    self.assertEqual(arguments[arguments.index('--precision') + 1], precision)
                for mode in ('production', 'lifecycle'):
                    arguments = commands[mode]
                    self.assertEqual(arguments[arguments.index('--precision') + 1], precision)
                    self.assertEqual(arguments[arguments.index('--model') + 1], model)


if __name__ == '__main__':
    unittest.main()
