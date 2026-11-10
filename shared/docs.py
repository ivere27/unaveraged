#!/usr/bin/env python3
"""Render canonical Markdown and benchmark JSON into an untracked static site."""
from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import posixpath
import re
import sys
from urllib.parse import quote, unquote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Explicit sources keep private/local files out of both the site and source ZIP.
DOCUMENTS = {
    'README.md': ('index.html', 'unaveraged', 'en'),
    'PRIVACY.md': ('privacy.html', 'Privacy policy', 'ko'),
    'benchmarks/README.md': ('benchmarks.html', 'Benchmarks', 'en'),
    'digit/README.md': ('digit.html', 'Digit', 'en'),
    'vqa/README.md': ('vqa.html', 'VQA', 'en'),
    'synth/README.md': ('synth.html', 'Synthetic receipts', 'en'),
    'STORE.md': ('store.html', 'Store listing', 'ko'),
}
ASSETS = {
    'docs/style.css': 'style.css',
    **{f'docs/assets/examples/{name}': f'assets/examples/{name}' for name in (
        'digit-receipt-ko.jpg', 'vqa-receipt-ko.jpg', 'sources.json')},
    'LICENSE': 'LICENSE', 'NOTICE': 'NOTICE',
    'benchmarks/results.json': 'benchmarks/results.json',
    'benchmarks/digit-comparison.json': 'benchmarks/digit-comparison.json',
}


def markdown_html(source):
    """One Markdown dialect for the website and extension privacy document."""
    try:
        import markdown
        from markdown.extensions.toc import slugify_unicode
    except ImportError as error:
        raise RuntimeError('Install docs/requirements.txt or run make docs') from error
    return markdown.markdown(source, extensions=['tables', 'fenced_code', 'toc'],
                             extension_configs={'toc': {'slugify': slugify_unicode}})


class SiteLinks(HTMLParser):
    """Resolve Markdown and authored HTML links without altering code examples."""
    def __init__(self, source, repo, source_ref='main'):
        super().__init__(convert_charrefs=False)
        self.source, self.repo, self.parts = source, repo, []
        self.source_ref = source_ref

    def url(self, value):
        url = urlsplit(value)
        if url.scheme or url.netloc or not url.path or url.path.startswith('/'):
            return value
        path = posixpath.normpath(posixpath.join(posixpath.dirname(self.source), unquote(url.path)))
        if path in DOCUMENTS:
            target = DOCUMENTS[path][0]
        elif path in ASSETS:
            target = ASSETS[path]
        elif path in ('example', 'example/index.html'):
            target = 'example/'
        else:
            kind = 'tree' if url.path.endswith('/') else 'blob'
            target = f'https://github.com/{self.repo}/{kind}/{quote(self.source_ref, safe="")}/{quote(path, safe="/")}'
        return urlunsplit(('', '', target, url.query, url.fragment))

    def start(self, tag, attrs, ending):
        if tag == 'table':
            self.parts.append('<div class="table-scroll">')
        attrs = [(name, self.url(value) if name in ('href', 'src') and value else value)
                 for name, value in attrs]
        self.parts.append('<' + tag + ''.join(
            f' {name}' if value is None else f' {name}="{html.escape(value, quote=True)}"'
            for name, value in attrs) + ending)

    def handle_starttag(self, tag, attrs):
        self.start(tag, attrs, '>')

    def handle_startendtag(self, tag, attrs):
        self.start(tag, attrs, ' />')

    def handle_endtag(self, tag):
        self.parts.append(f'</{tag}>')
        if tag == 'table':
            self.parts.append('</div>')

    def handle_data(self, data):
        self.parts.append(data)

    def handle_entityref(self, name):
        self.parts.append(f'&{name};')

    def handle_charref(self, name):
        self.parts.append(f'&#{name};')

    def handle_comment(self, data):
        self.parts.append(f'<!--{data}-->')


def page(title, body, active, language):
    links = [('index.html', 'Overview'), ('example/', 'Demo'),
             ('benchmarks.html', 'Benchmarks'), ('privacy.html', 'Privacy')]
    navigation = ' '.join(
        f'<a href="{url}"' + (' aria-current="page"' if url == active else '') + f'>{label}</a>'
        for url, label in links)
    return f'''<!doctype html>
<html lang="{language}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} · unaveraged</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <a class="skip" href="#main">Skip to content</a>
  <header><a class="brand" href="index.html">unaveraged</a><nav aria-label="Main">{navigation}</nav></header>
  <main id="main">{body}</main>
  <footer><a href="store.html">Store listing</a> · <a href="LICENSE">License</a> · <a href="NOTICE">Attribution</a></footer>
</body>
</html>
'''


def render(*, root=ROOT, repo='ivere27/unaveraged', source_ref='main'):
    from benchmarks.tables import render_tables
    tables = render_tables(json.loads((root/'benchmarks/results.json').read_text()),
                           json.loads((root/'benchmarks/digit-comparison.json').read_text()))
    output = {'.nojekyll': ''}
    for source, (target, title, language) in DOCUMENTS.items():
        text = (root/source).read_text()
        text = re.sub(r'<!-- benchmark:([a-z-]+) -->', lambda match: tables[match[1]], text)
        links = SiteLinks(source, repo, source_ref)
        links.feed(markdown_html(text))
        output[target] = page(title, ''.join(links.parts), target, language)
    return output


def write(output, *, root=ROOT, repo='ivere27/unaveraged', source_ref='main', check=False):
    """Write only known public docs/assets, also used inside atomic Pages staging."""
    output = Path(output)
    generated = render(root=root, repo=repo, source_ref=source_ref)
    files = {name: text.encode() for name, text in generated.items()}
    for source, target in ASSETS.items():
        data = (root/source).read_bytes()
        if source == 'docs/assets/examples/sources.json':
            # Resolve the same central notices from the site's shallower asset path.
            attribution = json.loads(data)
            for key, name in (('license_file', 'LICENSE'), ('vqa_notice_file', 'NOTICE')):
                attribution[key] = posixpath.relpath(name, posixpath.dirname(target))
            data = (json.dumps(attribution, ensure_ascii=False, indent=2) + '\n').encode()
        files[target] = data
    for name, data in files.items():
        path = output/name
        if check:
            if not path.is_file() or path.read_bytes() != data:
                raise ValueError(f'Outdated public document: {path}; run make docs')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'dist/pages')
    parser.add_argument('--repo', default='ivere27/unaveraged')
    parser.add_argument('--ref', dest='source_ref', default='main', help='Git revision for source links')
    parser.add_argument('--check', action='store_true', help='Verify generated output matches its sources')
    args = parser.parse_args()
    output = args.out.resolve()
    if output == ROOT or any(output.is_relative_to(ROOT/name) for name in
                             ('docs', 'example', 'shared', 'digit', 'vqa', 'synth', 'benchmarks')):
        parser.error('Choose a generated output directory, such as dist/pages')
    write(output, repo=args.repo, source_ref=args.source_ref, check=args.check)
    print(f'Public documentation: {output}')


if __name__ == '__main__':
    main()
