"""CPU gallery for matched MorphFlow runs with dataset and GLB-encoded inputs.

The metrics compare two model outputs. They do not score either against a
ground-truth morph. The portable preview contains images and measurements;
large GLBs, Gaussian payloads and latent tensors remain on the server.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
from html import escape
import json
import math
from pathlib import Path
import tempfile
from urllib.parse import unquote, urlsplit
import zipfile


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(nested, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            result[prefix] = value
    return result


def _alpha_dir(alpha):
    return ("alpha_%.6f" % float(alpha)).replace(".", "p")


def _rows(result):
    rows = result.get("alphas", []) if isinstance(result, dict) else []
    return rows if isinstance(rows, list) else []


def _complete(result, expected):
    rows = _rows(result)
    try:
        values = [float(row["alpha"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    return (len(values) == len(expected)
            and len(set(values)) == len(values)
            and all(any(abs(value - float(alpha)) < 1e-8 for value in values) for alpha in expected)
            and all(isinstance(row.get("geometry"), dict)
                    and isinstance(row.get("appearance"), dict) for row in rows))


def _artifact(root, folder, supplied=None, fallback=None):
    candidates = []
    if supplied:
        path = Path(supplied)
        candidates.extend([path] if path.is_absolute() else [root / path, folder / path])
    if fallback:
        candidates.append(folder / fallback)
    for path in candidates:
        absolute = path.resolve()
        if absolute.is_relative_to(root) and absolute.is_file():
            return absolute.relative_to(root).as_posix()
    return None


def _metric(row, *keys):
    values = _flatten(row)
    for key in keys:
        if key in values:
            return values[key]
    return None


def _number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def _center(result):
    rows = _rows(result)
    return min(rows, key=lambda row: abs(float(row["alpha"]) - 0.5)) if rows else None


def _reproduction(payload):
    if not isinstance(payload, dict):
        return None
    for key in ("exact_numeric_reproduction", "exact_reproduction", "exact_match", "all_exact", "exact_equal", "all_tensors_exact"):
        if isinstance(payload.get(key), bool):
            return payload[key]
    return None


def create_overview(root, manifest, completed):
    """Show one identical camera and full frame for every output/source tile."""
    from PIL import Image, ImageDraw
    from trellis_codec_overview import _font, _wrapped

    items = manifest["pairs"]
    if not items:
        return None
    size, left, gap, row_height, top = 280, 270, 12, 330, 150
    width, height = left + 4 * (size + gap) + gap, top + len(items) * row_height + 75
    canvas = Image.new("RGB", (width, height), "#edf0f4")
    draw = ImageDraw.Draw(canvas)
    title, heading, body, small = _font(26, True), _font(18, True), _font(15), _font(13)
    draw.text((18, 18), "MorphFlow full: condizioni dal dataset o da encoding GLB", font=title, fill="#172234")
    draw.text((18, 58), "Stessa camera, nessun riallineamento o riscalamento indipendente degli output.",
              font=body, fill="#475569")
    headings = ["Sorgente A", "Morph / SLat dataset", "Morph / SLat da GLB", "Sorgente B"]
    for column, label in enumerate(headings):
        draw.text((left + gap + column * (size + gap), 105), label, font=heading, fill="#172234")
    for index, item in enumerate(items):
        folder = root / "pairs" / item["pair_dir"]
        result = completed.get(item["pair_dir"])
        state = _center(result or {})
        y = top + index * row_height
        draw.rounded_rectangle((gap, y + 2, width - gap, y + row_height - 7), radius=7, fill="white")
        draw.text((22, y + 17), "Coppia %02d" % int(item.get("pair_id", index)), font=heading, fill="#172234")
        label = str(item["src1"]).replace("_", " ").replace("-", " ") + " → " + str(item["src2"]).replace("_", " ").replace("-", " ")
        for line, text in enumerate(_wrapped(draw, label, body, left - 35)[:12]):
            draw.text((22, y + 48 + line * 20), text, font=body, fill="#334155")
        paths = [
            _artifact(root, folder, fallback="comparison/src1_view_07.png"),
            _artifact(root, folder, (state or {}).get("paths", {}).get("reference_view_07"),
                      "comparison/%s/dataset_control_view_07.png" % (state or {}).get("alpha_dir", _alpha_dir(0.5))),
            _artifact(root, folder, (state or {}).get("paths", {}).get("encoded_view_07"),
                      "comparison/%s/glb_encoded_view_07.png" % (state or {}).get("alpha_dir", _alpha_dir(0.5))),
            _artifact(root, folder, fallback="comparison/src2_view_07.png"),
        ]
        for column, relative in enumerate(paths):
            x = left + gap + column * (size + gap)
            if relative:
                with Image.open(root / relative) as image:
                    # RGB is already composited on white; alpha is not applied twice.
                    tile = image.convert("RGB")
                    tile.thumbnail((size, size), Image.Resampling.LANCZOS)
                canvas.paste(tile, (x + (size - tile.width) // 2, y + 8 + (size - tile.height) // 2))
            else:
                draw.rectangle((x, y + 10, x + size, y + size), fill="#f4f6f8")
                draw.text((x + 75, y + 130), "In attesa", font=body, fill="#64748b")
        if state:
            alpha = float(state["alpha"])
            psnr = _metric(state, "appearance.psnr_foreground_union")
            iou = _metric(state, "appearance.silhouette_iou")
            draw.text((left + gap + size + gap + 3, y + size + 15), "alpha %.6f / controllo" % alpha,
                      font=body, fill="#334155")
            caption = "PSNR %s dB / IoU %s" % (_number(psnr, 2), _number(iou, 3))
            draw.text((left + gap + 2 * (size + gap) + 3, y + size + 15), caption, font=body, fill="#334155")
    foot = top + len(items) * row_height
    draw.text((18, foot + 6), "Ricostruzioni confrontate: %d/%d coppie. Immagine: vista 07 del morph alpha=0.5." % (
        len(completed), len(items)), font=small, fill="#475569")
    draw.text((18, foot + 28), "PSNR e IoU confrontano i due output su otto viste: misurano somiglianza, non qualità rispetto a un morph reale.",
              font=small, fill="#475569")
    target = root / "overview.png"
    temporary = root / "overview.tmp.png"
    canvas.save(temporary)
    temporary.replace(target)
    return target


def build_preview(root):
    """Package every local gallery reference while excluding binary 3D artifacts."""
    from trellis_codec_preview import _PreviewHTML

    parser = _PreviewHTML()
    parser.feed((root / "index.html").read_text())
    parser.close()
    page = "".join(parser.parts)
    notice = ('<aside class="note"><strong>Anteprima scaricabile.</strong> Estrai tutto lo ZIP e apri index.html. '
              'Sono incluse immagini e metriche; i GLB e i latenti rimangono sul server in '
              '<code>%s</code>.</aside>' % escape(str(root)))
    position = page.find("<h1")
    page = page[:position] + notice + page[position:] if position >= 0 else notice + page
    paths = set(parser.references)
    paths.update(name for name in ("manifest.json", "summary.json", "metrics.csv", "execution.json", "verification.json")
                 if (root / name).is_file())
    for item in _read(root / "manifest.json")["pairs"]:
        folder = root / "pairs" / item["pair_dir"]
        for path in folder.glob("dataset_control/*/control_reproduction.json"):
            paths.add(path.relative_to(root).as_posix())
    for relative in paths:
        path = (root / unquote(urlsplit(str(relative)).path)).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Preview reference leaves run directory: {relative}")
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".json", ".csv", ".html", ".css"}:
            raise ValueError(f"Unsupported preview artifact: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
    with tempfile.NamedTemporaryFile(prefix=".preview-", suffix=".zip.tmp", dir=root, delete=False) as stream:
        temporary = Path(stream.name)
    target = root / "preview.zip"
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            archive.writestr("index.html", page)
            for relative in sorted(paths):
                path = root / relative
                mode = zipfile.ZIP_STORED if path.suffix.lower() in {".png", ".jpg", ".jpeg"} else zipfile.ZIP_DEFLATED
                archive.write(path, arcname=relative, compress_type=mode, compresslevel=3)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _report(root):
    manifest = _read(root / "manifest.json")
    if not manifest or not isinstance(manifest.get("pairs"), list):
        raise ValueError("manifest.json must contain the frozen pair list")
    expected = manifest["alphas"]
    completed, pending, errors, csv_rows, historic = {}, [], [], [], []
    for index, item in enumerate(manifest["pairs"]):
        folder = root / "pairs" / item["pair_dir"]
        result = _read(folder / "result.json")
        if _complete(result, expected):
            completed[item["pair_dir"]] = result
            for row in _rows(result):
                values = {"pair_id": item["pair_id"], "pair_dir": item["pair_dir"],
                          "src1": item["src1"], "src2": item["src2"], "alpha": row["alpha"]}
                for group in ("geometry", "appearance", "structure", "slat"):
                    values.update(_flatten(row.get(group, {}), group))
                csv_rows.append(values)
        else:
            pending.append({"pair_id": item["pair_id"], "pair_dir": item["pair_dir"],
                            "status": _read(folder / "status.json", {"stage": "pending"})})
            if (folder / "error.json").is_file():
                errors.append({"pair_id": item["pair_id"], "error": _read(folder / "error.json")})
            elif (folder / "result.json").is_file():
                errors.append({"pair_id": item["pair_id"], "error": "Pair result has an incomplete or invalid alpha/metric list"})
        for alpha in expected:
            path = folder / "dataset_control" / _alpha_dir(alpha) / "control_reproduction.json"
            if path.is_file():
                payload = _read(path)
                historic.append({"pair_id": item["pair_id"], "alpha": alpha,
                                 "exact": _reproduction(payload), "path": path.relative_to(root).as_posix()})
    columns = ["pair_id", "pair_dir", "src1", "src2", "alpha"] + sorted({key for row in csv_rows for key in row}
        - {"pair_id", "pair_dir", "src1", "src2", "alpha"})
    temporary = root / "metrics.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)
    temporary.replace(root / "metrics.csv")
    numeric = sorted({key for row in csv_rows for key, value in row.items()
                      if isinstance(value, (float, int)) and key not in {"pair_id", "alpha"}})
    means = {key: sum(float(row[key]) for row in csv_rows if key in row) / sum(key in row for row in csv_rows)
             for key in numeric}
    summary = {
        "status": "complete" if len(completed) == len(manifest["pairs"]) else "incomplete",
        "completed_pairs": len(completed), "expected_pairs": len(manifest["pairs"]),
        "completed_alpha_comparisons": len(csv_rows), "expected_alpha_comparisons": len(manifest["pairs"]) * len(expected),
        "means": means, "pending": pending, "failures": errors,
        "historical_reproduction": {"checked": len(historic), "exact": sum(row["exact"] is True for row in historic),
                                    "different": sum(row["exact"] is False for row in historic),
                                    "unknown": sum(row["exact"] is None for row in historic), "states": historic},
        "interpretation": "Distances between matched model outputs with dataset versus GLB-encoded conditioning; not quality against ground-truth morphs.",
        "comparison_scope": "Both SS and SLat flows consume only their branch's source SLat and coordinates. The encoded branch does not reuse dataset source features.",
        "noise_caveat": "Equal SLat seed reproduces a noise sequence; when SS coordinates differ, that sequence is assigned to different sparse voxels.",
        "sampling": manifest.get("sampling"), "checkpoints": manifest.get("checkpoints"),
    }
    _write(root / "summary.json", summary)
    create_overview(root, manifest, completed)
    metric_columns = [("Chamfer ↓", "geometry.chamfer_l1", 6),
                      ("IoU silhouette ↑", "appearance.silhouette_iou", 4),
                      ("PSNR oggetto ↑", "appearance.psnr_foreground_union", 2),
                      ("IoU voxel SS ↑", "structure.occupancy_iou", 4),
                      ("RMSE SS ↓", "structure.ss_rmse", 5)]
    aggregate_table = '<table><tr><th>Confronti</th>' + ''.join('<th>%s</th>' % escape(label) for label, _, _ in metric_columns) + '</tr>'
    aggregate_table += '<tr><td>%d</td>' % len(csv_rows) + ''.join('<td>%s</td>' % _number(means.get(key), digits)
        for _, key, digits in metric_columns) + '</tr></table>'
    cards = []
    for index, item in enumerate(manifest["pairs"]):
        name = item["pair_dir"]
        folder = root / "pairs" / name
        result = completed.get(name)
        links = []
        for label, relative in [("GLB sorgente A", "assets/%s/source.glb" % item["src1"]),
                                ("GLB sorgente B", "assets/%s/source.glb" % item["src2"])]:
            if (root / relative).is_file():
                links.append('<a href="%s">%s</a>' % (escape(relative, quote=True), label))
        title = '<h2>Coppia %02d: %s → %s</h2>' % (item["pair_id"], escape(item["src1"]), escape(item["src2"]))
        content = title + '<p>' + ' | '.join(links) + '</p>'
        if result:
            result_path = "pairs/%s/result.json" % name
            content += '<p><a href="%s">Metriche complete della coppia</a></p>' % escape(result_path, quote=True)
            trajectory = _artifact(root, folder, result.get("trajectory_png"), "comparison.png")
            if trajectory:
                content += '<a href="%s"><img loading="lazy" src="%s" alt="Confronto delle traiettorie"></a>' % (escape(trajectory, quote=True), escape(trajectory, quote=True))
            content += '<table><tr><th>alpha</th><th>GLB: SLat dataset</th><th>GLB: SLat da encoding</th>'
            content += ''.join('<th>%s</th>' % escape(label) for label, _, _ in metric_columns) + '</tr>'
            for row in _rows(result):
                directory = row.get("alpha_dir", _alpha_dir(row["alpha"]))
                cells = ['<td>%.6f</td>' % row["alpha"]]
                for branch in ("dataset_control", "glb_encoded"):
                    options = []
                    for filename, label in (("prediction.glb", "Colori ai vertici"),
                                            ("prediction_textured.glb", "Texture ufficiale")):
                        relative = "pairs/%s/%s/%s/%s" % (name, branch, directory, filename)
                        if (root / relative).is_file():
                            options.append('<a href="%s">%s</a>' % (escape(relative, quote=True), label))
                    cells.append('<td>%s</td>' % (' / '.join(options) or "In attesa"))
                cells += ['<td>%s</td>' % _number(_metric(row, key), digits) for _, key, digits in metric_columns]
                content += '<tr>' + ''.join(cells) + '</tr>'
            content += '</table>'
        else:
            content += '<p class="muted">Confronto in corso: le metriche appariranno dopo il completamento dei cinque alpha.</p>'
        cards.append('<article>' + content + '</article>')
    sampling = manifest.get("sampling", {})
    ss_meta, slat_meta = manifest.get("checkpoints", {}).get("ss", {}), manifest.get("checkpoints", {}).get("slat", {})
    ss_epoch, slat_epoch = ss_meta.get("epoch", 17), slat_meta.get("epoch", 12)
    page = '<!doctype html><html lang="it"><meta charset="utf-8"><title>MorphFlow: input da GLB o dataset</title>'
    page += '<style>body{font:16px system-ui;margin:24px;background:#edf0f4;color:#172234}article,.note{background:white;padding:20px;margin:20px 0;border-radius:8px}img{max-width:100%}a{color:#145da8}table{border-collapse:collapse;background:white;margin:18px 0}td,th{padding:9px;border:1px solid #cbd5e1;text-align:left}h2{font-size:18px;overflow-wrap:anywhere}.muted{color:#64748b}code{overflow-wrap:anywhere}</style>'
    page += '<h1>MorphFlow full: %d/%d coppie completate</h1>' % (len(completed), len(manifest["pairs"]))
    page += '<p><a href="overview.png">Tavola riassuntiva</a> | <a href="preview.zip">Scarica anteprima</a> | <a href="metrics.csv">Metriche CSV</a> | <a href="summary.json">Riepilogo JSON</a> | <a href="manifest.json">Protocollo e coppie</a></p>'
    page += '<p>Controllo: SLat sorgenti del dataset → SS → SLat → decoding. Prova: GLB dei due oggetti → encoding multivista → nuovi SLat sorgenti → gli stessi SS e SLat → decoding. L’output della prova usa interamente le condizioni ricavate dai GLB.</p>'
    page += '<p>Checkpoint full SS epoca %s, SLat epoca %s; CFG %s / %s; %s step SS e %s step SLat; precisione %s. Alpha e seed sono identici tra i due percorsi. Non viene applicato alcun trasferimento dei colori sorgente agli output del morphing.</p>' % tuple(map(escape, map(str, (
        ss_epoch, slat_epoch, sampling.get("cfg_ss", 5), sampling.get("cfg_slat", 1.5),
        sampling.get("ss_steps", 25), sampling.get("slat_steps", 25), sampling.get("mixed_precision", "bf16")))))
    page += '<p>La prima riga di ogni confronto mostra gli output dagli SLat dataset; la seconda usa gli SLat ottenuti dai GLB. Gli estremi sono i due oggetti sorgente; le cinque colonne interne sono i morph agli alpha salvati.</p>'
    page += aggregate_table
    page += '<p>Le distanze misurano la differenza tra i due output, non la qualità rispetto a un morph reale. PSNR e IoU sono calcolati su otto viste comuni; la tavola riassuntiva mostra una sola vista ad alpha=0,5. Le geometrie non vengono riallineate o riscalate separatamente.</p>'
    history = summary["historical_reproduction"]
    page += '<p>Verifica del controllo contro i risultati storici: %d stati controllati, %d identici, %d differenti, %d senza esito interpretabile. I dettagli rimangono nel riepilogo JSON.</p>' % (history["checked"], history["exact"], history["different"], history["unknown"])
    page += '<p>Lo stesso seed SLat genera la stessa sequenza casuale; quando SS produce coordinate diverse, i numeri sono assegnati a voxel diversi. Il confronto misura quindi l’effetto complessivo del ciclo GLB → condizioni → morphing.</p>'
    page += ''.join(cards) + '</html>'
    temporary = root / "index.html.tmp"
    temporary.write_text(page)
    temporary.replace(root / "index.html")
    build_preview(root)
    return summary


def report(runroot):
    """Refresh summary, CSV, HTML, overview and ZIP using complete pair commits."""
    root = Path(runroot).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _report(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runroot", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(report(args.runroot), indent=2))


if __name__ == "__main__":
    main()
