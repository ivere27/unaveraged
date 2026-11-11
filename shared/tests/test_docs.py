"""Public HTML is derived from authored Markdown and aggregate JSON only."""
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

from shared import build, docs


ROOT = Path(__file__).resolve().parents[2]


class Links(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.urls, self.ids = [], set()
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ('href', 'src') and value:
                self.urls.append(value)
            elif name == 'id':
                self.ids.add(value)


class DocsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/'source'
        self.output = Path(self.temporary.name)/'site'
        for name in set(docs.DOCUMENTS) | set(docs.ASSETS):
            target = self.root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)

    def test_clean_authored_sources_generate_complete_linked_site(self):
        self.assertFalse(list((self.root/'docs').rglob('*.html')))
        (self.root/'docs/private-notes.md').write_text('private source must not publish')
        docs.write(self.output, root=self.root)
        # The separate Pages staging step supplies the interactive demo.
        (self.output/'example').mkdir()
        (self.output/'example/index.html').write_text('<main>Demo fixture</main>')
        self.assertEqual(len(list(self.output.glob('*.html'))), 7)
        self.assertTrue((self.output/'.nojekyll').exists())
        self.assertFalse((self.output/'private-notes.md').exists())
        for directory in (self.root/'docs/assets/examples', self.output/'assets/examples'):
            attribution = json.loads((directory/'sources.json').read_text())
            for key, name in (('license_file', 'LICENSE'), ('vqa_notice_file', 'NOTICE')):
                self.assertEqual((directory/attribution[key]).read_bytes(), (self.root/name).read_bytes())
        for page in self.output.glob('*.html'):
            parsed = Links(page.read_text())
            for value in parsed.urls:
                url = urlsplit(value)
                if url.scheme or url.netloc:
                    continue
                target = page.parent/unquote(url.path) if url.path else page
                if target.is_dir():
                    target /= 'index.html'
                with self.subTest(page=page.name, link=value):
                    self.assertTrue(target.is_file(), f'Missing local target: {target}')
                    if url.fragment:
                        self.assertIn(unquote(url.fragment), Links(target.read_text()).ids)

    def test_markdown_and_authored_html_rewrite_links_without_changing_code(self):
        (self.root/'digit/README.md').write_text('''# 한국어 제목

[정책](../PRIVACY.md#보관과-삭제)

| Input | Value |
| --- | --- |
| Pixels | 672 |

```html
<img src="../docs/assets/examples/digit-receipt-ko.jpg">
```

<a href="../vqa/README.md#model-and-runtime"><img src="../docs/assets/examples/digit-receipt-ko.jpg" alt="합성"></a>

[Source](check.py?view=1#details)
''')
        page = docs.render(root=self.root, repo='owner/project')['digit.html']
        self.assertIn('<h1 id="한국어-제목">한국어 제목</h1>', page)
        self.assertIn('href="privacy.html#보관과-삭제"', page)
        self.assertIn('href="vqa.html#model-and-runtime"', page)
        self.assertIn('src="assets/examples/digit-receipt-ko.jpg"', page)
        self.assertIn('<table>', page)
        self.assertIn('<td>672</td>', page)
        self.assertIn('<pre><code class="language-html">', page)
        self.assertIn('<img src="../docs/assets/examples/digit-receipt-ko.jpg">', unescape(page))
        self.assertIn('https://github.com/owner/project/blob/main/digit/check.py?view=1#details', page)
        for source_ref in ('master', 'a1' * 20):
            docs.write(self.output, root=self.root, repo='owner/project', source_ref=source_ref)
            rendered = (self.output/'digit.html').read_text()
            self.assertIn(f'https://github.com/owner/project/blob/{source_ref}/digit/check.py?view=1#details', rendered)
            self.assertIn('href="privacy.html#보관과-삭제"', rendered)

    def test_privacy_changes_reach_site_and_packaged_policy_from_one_source(self):
        policy = '# 정책 변경\n\n## 새 보관 규칙\n\n개정된 **로컬 정책** · revision-unique.\n'
        (self.root/'PRIVACY.md').write_text(policy)
        site = docs.render(root=self.root)['privacy.html']
        extension = build.privacy_page(root=self.root)
        for page in (site, extension):
            self.assertIn('revision-unique', page)
            self.assertIn('<strong>로컬 정책</strong>', page)
            self.assertIn('id="새-보관-규칙"', page)

    def test_benchmark_tables_follow_both_json_sources(self):
        before = docs.render(root=self.root)['benchmarks.html']
        results_path = self.root/'benchmarks/results.json'
        results = json.loads(results_path.read_text())
        results['models']['vqa']['latency']['w4a8']['complete_answer_median_ms'] = 12345.67
        results_path.write_text(json.dumps(results))
        comparison_path = self.root/'benchmarks/digit-comparison.json'
        comparison = json.loads(comparison_path.read_text())
        comparison['models'][0]['latency']['chrome']['median_ms'] = 76543.21
        comparison_path.write_text(json.dumps(comparison))
        after = docs.render(root=self.root)['benchmarks.html']
        for measurement in ('12345.67', '76543.21'):
            self.assertNotIn(measurement, before)
            self.assertIn(f'<td>{measurement}</td>', after)
        self.assertNotIn('<!-- benchmark:', after)

    def test_check_detects_stale_source_rendering_and_copied_assets(self):
        docs.write(self.output, root=self.root)
        docs.write(self.output, root=self.root, check=True)
        policy = self.root/'PRIVACY.md'
        policy.write_text(policy.read_text() + '\nNew policy revision.\n')
        with self.assertRaisesRegex(ValueError, 'privacy.html'):
            docs.write(self.output, root=self.root, check=True)
        docs.write(self.output, root=self.root)
        (self.output/'style.css').write_text('stale stylesheet')
        with self.assertRaisesRegex(ValueError, 'style.css'):
            docs.write(self.output, root=self.root, check=True)


if __name__ == '__main__':
    unittest.main()
