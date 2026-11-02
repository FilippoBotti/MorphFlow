"""Package a portable codec gallery with images and measurements, using stdlib.

The server's index and large GLB/latent/render artifacts are left untouched.
Extract preview.zip and open its root index.html in any desktop browser.
"""

from __future__ import annotations

import argparse
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote, urlsplit
import zipfile


ROOT_FILES = ("overview.png", "metrics.csv", "summary.json", "manifest.json", "execution.json")
SMALL_JSON_BYTES = 1024 * 1024
PREVIEW_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".json", ".csv", ".css"}


def _local_path(url):
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return unquote(parsed.path)


class _PreviewHTML(HTMLParser):
    """Preserve the gallery while replacing links to server-only artifacts."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.parts = []
        self.references = set()
        self.replaced_anchor = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        target = _local_path(attributes.get("href", "")) if tag == "a" else None
        if tag == "a" and target and Path(target).suffix.lower() in {".glb", ".zip"}:
            self.replaced_anchor = True
            label = "GLB sul server" if Path(target).suffix.lower() == ".glb" else "Archivio già scaricato"
            self.parts.append(f'<span class="server-file">{label}: ')
            return
        for attribute in ("src", "href"):
            value = attributes.get(attribute)
            local = _local_path(value) if value else None
            if local:
                self.references.add(local)
        self.parts.append(self.get_starttag_text())

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag == "a" and self.replaced_anchor:
            self.replaced_anchor = False
            self.parts.append("</span>")
        else:
            self.parts.append(f"</{tag}>")

    def handle_data(self, data):
        self.parts.append(data)

    def handle_entityref(self, name):
        self.parts.append(f"&{name};")

    def handle_charref(self, name):
        self.parts.append(f"&#{name};")

    def handle_decl(self, declaration):
        self.parts.append(f"<!{declaration}>")

    def handle_comment(self, data):
        self.parts.append(f"<!--{data}-->")


def build_preview_zip(runroot) -> Path:
    """Atomically create ``runroot/preview.zip`` and return its absolute Path.

    Include the current gallery, its linked images, overview and root reports,
    asset results, and comparison JSON files up to 1 MiB. No GLBs, tensors,
    feature caches or view render directories are traversed or copied.
    """
    root = Path(runroot).expanduser().resolve()
    source_html = (root / "index.html").read_text(encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    files = {}

    def include(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Preview reference leaves the run directory: {relative}")
        if path.suffix.lower() not in PREVIEW_SUFFIXES:
            raise ValueError(f"Unsupported preview artifact: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing gallery artifact: {path}")
        files[path.relative_to(root).as_posix()] = path

    for filename in ROOT_FILES:
        include(filename)
    has_verification = (root / "verification.json").is_file()
    if has_verification:
        include("verification.json")
    for item in manifest["assets"]:
        asset = Path(f"{item['index']:02d}_{item['asset']}")
        for relative in ("comparison.png", "result.json", "source_texture/comparison.png", "source_texture/result.json"):
            candidate = asset / relative
            if (root / candidate).is_file():
                include(candidate)
        for branch in ("legacy_50", "official_150", "source_texture"):
            for comparison in ("comparison", "comparison_vertex"):
                candidate = asset / branch / comparison / "metrics.json"
                path = root / candidate
                if path.is_file() and path.stat().st_size <= SMALL_JSON_BYTES:
                    include(candidate)

    parser = _PreviewHTML()
    parser.feed(source_html)
    parser.close()
    for reference in sorted(parser.references):
        include(reference)
    html = "".join(parser.parts)
    introduction = (
        '<aside style="background:#e8f2ff;border:1px solid #9abce4;padding:18px;margin:20px 0">'
        '<h2>Anteprima: immagini e metriche</h2>'
        '<p>Estrai tutto il contenuto di <strong>preview.zip</strong> e apri '
        '<strong>index.html</strong> nel browser. Questo archivio contiene immagini, '
        'tabelle e risultati JSON. I GLB completi rimangono sul server nella cartella '
        f'<code>{escape(str(root))}</code>; le etichette «GLB sul server» indicano quei file.</p>'
        '<p><a href="overview.png">Tavola riassuntiva</a> | '
        '<a href="metrics.csv">Metriche CSV</a> | <a href="summary.json">Riepilogo JSON</a> | '
        '<a href="manifest.json">Selezione degli asset</a> | '
        '<a href="execution.json">Dettagli di esecuzione</a>'
        + (' | <a href="verification.json">Verifica dei GLB sul server</a>' if has_verification else '')
        + '</p></aside>'
    )
    body = re.search(r"<body\b[^>]*>", html, re.IGNORECASE)
    heading = re.search(r"<h1\b", html, re.IGNORECASE)
    position = body.end() if body else heading.start() if heading else len(html)
    html = html[:position] + introduction + html[position:]

    destination = root / "preview.zip"
    with tempfile.NamedTemporaryFile(prefix=".preview-", suffix=".zip.tmp", dir=root, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            archive.writestr("index.html", html.encode("utf-8"))
            for relative, path in sorted(files.items()):
                compression = zipfile.ZIP_STORED if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else zipfile.ZIP_DEFLATED
                archive.write(path, arcname=relative, compress_type=compression, compresslevel=3)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runroot", type=Path, help="Benchmark directory containing index.html and manifest.json")
    args = parser.parse_args(argv)
    print(build_preview_zip(args.runroot))


if __name__ == "__main__":
    main()
