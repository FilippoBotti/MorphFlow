"""CPU-only, shared-crop overview of the fixed-view TRELLIS codec benchmark."""

import json
from pathlib import Path


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _font(size, bold=False):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [Path("/usr/share/fonts/dejavu") / name,
                  Path("/usr/share/fonts/truetype/dejavu") / name,
                  Path("/usr/share/fonts/truetype/liberation2") /
                  ("LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf")]
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    return ImageFont.load_default()


def _wrapped(draw, text, font, width):
    """Wrap labels by actual pixel width, also splitting long asset tokens."""
    lines, line = [], ""
    for word in text.split():
        while word and draw.textbbox((0, 0), word, font=font)[2] > width:
            cut = 1
            while cut < len(word) and draw.textbbox((0, 0), word[:cut + 1], font=font)[2] <= width:
                cut += 1
            if line:
                lines.append(line)
                line = ""
            lines.append(word[:cut])
            word = word[cut:]
        candidate = (line + " " + word).strip()
        if line and draw.textbbox((0, 0), candidate, font=font)[2] > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def _source_bbox(path):
    from PIL import Image

    if not path.is_file():
        return None, None
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        alpha = rgba.getchannel("A").point(lambda value: 255 if value > 127 else 0)
        bounds = alpha.getbbox()
        if bounds is None:
            bounds = (0, 0, *rgba.size)
        left, top, right, bottom = bounds
        padding = max(8, int(round(max(right - left, bottom - top) * 0.08)))
        bounds = (max(0, left - padding), max(0, top - padding),
                  min(rgba.width, right + padding), min(rgba.height, bottom + padding))
        return bounds, rgba.size


def _tile(path, bounds, source_size, size):
    from PIL import Image

    if not path.is_file() or bounds is None:
        return None
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        if rgba.size != source_size:
            # Comparing a differently framed rendering would silently change
            # the crop semantics. Keep that cell visibly unavailable instead.
            return None
        # The PNG RGB is already composited over white by the renderer. Dropping
        # alpha avoids applying it twice at partially transparent edges.
        crop = rgba.convert("RGB").crop(bounds)
    factor = min(size / crop.width, size / crop.height)
    crop = crop.resize((max(1, round(crop.width * factor)),
                        max(1, round(crop.height * factor))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return canvas


def _psnr(path):
    data = _read_json(path)
    value = (data or {}).get("appearance", {}).get("psnr_foreground_union")
    return "PSNR oggetto %.2f dB" % value if isinstance(value, (int, float)) else ""


def create_overview(runroot):
    """Write overview.png for all manifest assets, retaining pending rows.

    Every row uses the exact same source-derived crop for all four methods.
    The displayed view is #07: yaw 292.5 degrees, elevation 32 degrees. Metrics
    captions are the foreground-union PSNR averaged over all eight held-out views.
    Returns the absolute PNG path, or None when no manifest assets are available.
    """
    from PIL import Image, ImageDraw

    runroot = Path(runroot).expanduser().resolve()
    manifest = _read_json(runroot / "manifest.json")
    if not manifest or not manifest.get("assets"):
        return None
    assets = manifest["assets"]
    image_size, label_width, gap = 300, 245, 12
    row_height, header_height, footer_height = 344, 154, 88
    width = label_width + 4 * (image_size + gap) + gap
    height = header_height + row_height * len(assets) + footer_height
    canvas = Image.new("RGB", (width, height), "#edf0f4")
    draw = ImageDraw.Draw(canvas)
    title_font, heading_font = _font(27, True), _font(19, True)
    body_font, small_font = _font(17), _font(14)
    draw.text((18, 15), "TRELLIS: confronto su %d asset del test v4" % len(assets),
              font=title_font, fill="#172234")
    draw.text((18, 55), "Stessa vista e stesso ritaglio per ogni riga; nessun riallineamento delle ricostruzioni.",
              font=body_font, fill="#475569")
    headings = ["Decoding diretto", "Ricostruzione / 50 viste", "Ricostruzione / 150 viste",
                "Aspetto dalla sorgente*"]
    for column, heading in enumerate(headings):
        x = label_width + gap + column * (image_size + gap)
        for line_index, line in enumerate(_wrapped(draw, heading, heading_font, image_size)):
            draw.text((x + 4, 102 + line_index * 23), line, font=heading_font, fill="#172234")
    completed = 0
    source_texture_completed = 0
    for row_index, item in enumerate(assets):
        name = item["asset"]
        index = item.get("index", row_index)
        folder = runroot / ("%02d_%s" % (index, name))
        if (folder / "result.json").is_file():
            completed += 1
        y = header_height + row_index * row_height
        draw.rounded_rectangle((gap, y + 2, width - gap, y + row_height - 6),
                               radius=7, fill="white")
        label = name.replace("_", " ", 1).replace("-", " ")
        lines = _wrapped(draw, label, body_font, label_width - 2 * gap)
        draw.text((24, y + 26), "%02d" % index, font=heading_font, fill="#172234")
        for line_index, line in enumerate(lines[:7]):
            draw.text((24, y + 59 + line_index * 23), line, font=body_font, fill="#334155")
        source = folder / "official_150/comparison/source_view_07.png"
        bounds, source_size = _source_bbox(source)
        paths = [source, folder / "legacy_50/comparison/recon_view_07.png",
                 folder / "official_150/comparison/recon_view_07.png",
                 folder / "source_texture/comparison/recon_view_07.png"]
        if paths[-1].is_file() and (folder / "source_texture/comparison/metrics.json").is_file():
            source_texture_completed += 1
        captions = ["Riferimento",
                    _psnr(folder / "legacy_50/comparison/metrics.json"),
                    _psnr(folder / "official_150/comparison/metrics.json"),
                    _psnr(folder / "source_texture/comparison/metrics.json")]
        for column, (path, caption) in enumerate(zip(paths, captions)):
            x = label_width + gap + column * (image_size + gap)
            tile = _tile(path, bounds, source_size, image_size)
            if tile is not None:
                canvas.paste(tile, (x, y + 5))
                draw.text((x + 6, y + image_size + 13), caption,
                          font=body_font, fill="#334155")
            else:
                draw.rectangle((x + 6, y + 12, x + image_size - 6, y + image_size - 4),
                               fill="#f3f5f8")
                draw.text((x + 95, y + 142), "In attesa", font=body_font, fill="#64748b")
    footer_y = header_height + row_height * len(assets) + 8
    draw.text((18, footer_y),
              "* Usa il GLB sorgente per la texture; geometria ricostruita invariata. Non e' decoding dal solo latente.",
              font=small_font, fill="#475569")
    draw.text((18, footer_y + 24),
              "Vista: yaw 292.5 / elevazione 32. PSNR oggetto: media su 8 viste separate da quelle di fitting.",
              font=small_font, fill="#475569")
    draw.text((18, footer_y + 48),
              "Ricostruzioni complete: %d/%d asset. Texture dalla sorgente: %d/%d." % (
                  completed, len(assets), source_texture_completed, len(assets)),
              font=small_font, fill="#475569")
    output = runroot / "overview.png"
    temporary = runroot / "overview.tmp.png"
    canvas.save(temporary)
    temporary.replace(output)
    return str(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runroot", type=Path)
    args = parser.parse_args()
    print(create_overview(args.runroot))
