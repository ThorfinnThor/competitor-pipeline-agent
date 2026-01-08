import os
import re
import json
import time
import csv
import hashlib
import datetime
from typing import Optional, List, Dict, Any, Tuple

import yaml
import requests

import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

from google import genai
from google.genai import types  # IMPORTANT for image parts

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from xml.sax.saxutils import escape as xml_escape


# =============================
# Utilities
# =============================
DATE_JSON_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")
DATE_PDF_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.pdf$")


def utc_today_str() -> str:
    return datetime.datetime.utcnow().date().isoformat()


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def safe_json_load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_json_dump(path: str, obj: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_text_dump(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def http_get(url: str, headers: Optional[dict] = None, timeout: int = 60) -> requests.Response:
    return requests.get(url, headers=headers or {}, timeout=timeout)


def http_head_ok(url: str, timeout: int = 30) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def deep_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def program_count(snapshot: Optional[dict]) -> int:
    if not snapshot:
        return 0
    return len(snapshot.get("programs", []) or [])


def find_latest_valid_snapshot(snap_dir: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Recover the most recent dated snapshot that has programs > 0.
    Protects you if latest.json was overwritten by a failed run.
    """
    if not os.path.isdir(snap_dir):
        return None, None

    dated = [fn for fn in os.listdir(snap_dir) if DATE_JSON_RE.match(fn)]
    dated.sort(reverse=True)

    for fn in dated:
        path = os.path.join(snap_dir, fn)
        s = safe_json_load(path)
        if program_count(s) > 0:
            return s, path

    return None, None


def get_snapshot_date_from_path(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\.(json|pdf)$", base)
    return m.group(1) if m else None


def best_prev_pdf_hash(prev_snapshot: Optional[dict], sources_dir: str, snap_dir: str) -> Optional[str]:
    meta = (prev_snapshot or {}).get("_meta", {}) or {}
    if meta.get("source_sha256"):
        return meta["source_sha256"]

    stored = meta.get("stored_pdf_path")
    if stored and os.path.exists(stored):
        try:
            return sha256_file(stored)
        except Exception:
            pass

    date_guess = meta.get("run_date_utc")
    if not date_guess:
        _, path = find_latest_valid_snapshot(snap_dir)
        if path:
            date_guess = get_snapshot_date_from_path(path)

    if date_guess:
        candidate = os.path.join(sources_dir, f"{date_guess}.pdf")
        if os.path.exists(candidate):
            try:
                return sha256_file(candidate)
            except Exception:
                pass

    if os.path.isdir(sources_dir):
        pdfs = [fn for fn in os.listdir(sources_dir) if DATE_PDF_RE.match(fn)]
        pdfs.sort(reverse=True)
        for fn in pdfs:
            p = os.path.join(sources_dir, fn)
            try:
                return sha256_file(p)
            except Exception:
                continue

    return None


# =============================
# J&J Q4 CDN auto-discovery
# =============================
def quarter_of_date(d: datetime.date) -> int:
    return (d.month - 1) // 3 + 1


def prev_quarter(year: int, q: int) -> Tuple[int, int]:
    if q == 1:
        return year - 1, 4
    return year, q - 1


def discover_latest_jnj_pipeline_pdf(q4cdn_base: str, filename_prefix: str, lookback_quarters: int) -> str:
    today = datetime.datetime.utcnow().date()
    y = today.year
    q = quarter_of_date(today)

    candidates: List[str] = []
    for _ in range(max(1, lookback_quarters)):
        yy = y % 100
        candidates.append(f"{q4cdn_base}/{y}/q{q}/{filename_prefix}-{q}Q{yy:02d}.pdf")
        y, q = prev_quarter(y, q)

    for url in candidates:
        if http_head_ok(url):
            return url

    for url in candidates:
        try:
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code == 200:
                return url
        except Exception:
            pass

    raise RuntimeError("Could not auto-discover a valid J&J pipeline PDF in the lookback window.")


# =============================
# PDF extraction + OCR fallback
# =============================
def pdf_to_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts: List[str] = []
    for i in range(doc.page_count):
        parts.append(doc.load_page(i).get_text("text"))
    return "\n".join(parts)


def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    img = pil_img.convert("L")
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    return img


def ocr_pdf_pages(pdf_bytes: bytes, max_pages: int = 8, dpi: int = 220) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = min(doc.page_count, max_pages)

    texts: List[str] = []
    for i in range(n_pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi)
        mode = "RGB" if pix.alpha == 0 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        img = preprocess_for_ocr(img)
        t = pytesseract.image_to_string(img, config="--psm 6").strip()
        if t:
            texts.append(f"\n\n===== OCR PAGE {i+1} =====\n{t}")

    return "\n".join(texts)


def maybe_add_ocr(pdf_bytes: bytes, text_extracted: str, ocr_max_pages: int, ocr_dpi: int) -> str:
    if len((text_extracted or "").strip()) >= 5000:
        return text_extracted
    return (text_extracted or "") + "\n\n" + ocr_pdf_pages(pdf_bytes, max_pages=ocr_max_pages, dpi=ocr_dpi)


# =============================
# Gemini helpers
# =============================
def gemini_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY env var.")
    return genai.Client(api_key=key)


def _try_parse_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def gemini_generate_text(models: List[str], prompt: str, max_attempts_per_model: int = 2) -> str:
    client = gemini_client()
    last_err: Optional[Exception] = None

    for m in models:
        for attempt in range(1, max_attempts_per_model + 1):
            try:
                resp = client.models.generate_content(model=m, contents=prompt)
                return (resp.text or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(2 * attempt)

    raise RuntimeError(f"All Gemini attempts failed. Last error: {last_err}")


def gemini_generate_vision(models: List[str], prompt: str, image_bytes: bytes, max_attempts_per_model: int = 2) -> str:
    """
    Inline image prompting (image bytes + text prompt).
    """
    client = gemini_client()
    last_err: Optional[Exception] = None

    for m in models:
        for attempt in range(1, max_attempts_per_model + 1):
            try:
                img_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                # Best practice: image first, then text prompt
                resp = client.models.generate_content(model=m, contents=[img_part, prompt])
                return (resp.text or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(2 * attempt)

    raise RuntimeError(f"All Gemini vision attempts failed. Last error: {last_err}")


# =============================
# Vision pipeline extraction (phase via colors/layout)
# =============================
def render_page_png(pdf_doc: fitz.Document, page_index: int, dpi: int) -> bytes:
    page = pdf_doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def llm_extract_pipeline_from_page_image(company_name: str, page_no_1based: int, png_bytes: bytes, models: List[str]) -> dict:
    prompt = f"""
You are a pharma competitive intelligence analyst.

TASK:
Extract all pipeline programs visible on this page image for {company_name}.
Phase is encoded visually (color and/or layout). Infer the phase from the page itself.
If you cannot determine phase with confidence, set it to null (do NOT guess).

Return ONLY valid JSON with this schema:
{{
  "as_of_date": string | null,
  "programs": [
    {{
      "asset": string,
      "indication": string | null,
      "phase": string | null,
      "trial_or_program": string | null,
      "partner": string | null,
      "notes": string | null,
      "source_page": {page_no_1based}
    }}
  ]
}}

Phase must be one of:
"Preclinical", "Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Registration", "Approved"

Rules:
- Do not invent assets.
- Include every asset tile/entry you can read.
- If the page shows an "as of" date (e.g., "as of October 14, 2025"), capture it as as_of_date.
""".strip()

    out = gemini_generate_vision(models=models, prompt=prompt, image_bytes=png_bytes, max_attempts_per_model=2)
    parsed = _try_parse_json(out)
    if parsed is None:
        return {
            "as_of_date": None,
            "programs": [],
            "_llm_parse_error": "Could not parse JSON from vision output",
            "_llm_raw_output": out[:8000],
        }
    parsed.setdefault("as_of_date", None)
    parsed.setdefault("programs", [])
    # enforce page number on all rows
    for p in parsed["programs"]:
        p["source_page"] = page_no_1based
    return parsed


def llm_extract_pipeline_vision(company_name: str, pdf_bytes: bytes, models: List[str], max_pages: int, dpi: int) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = min(doc.page_count, max_pages)

    all_programs: List[dict] = []
    as_of: Optional[str] = None

    for i in range(n):
        png = render_page_png(doc, i, dpi=dpi)
        page_res = llm_extract_pipeline_from_page_image(company_name, i + 1, png, models)
        if not as_of and page_res.get("as_of_date"):
            as_of = page_res.get("as_of_date")
        all_programs.extend(page_res.get("programs", []) or [])

    # de-dup by asset+indication
    def canon(s: Optional[str]) -> str:
        if not s:
            return ""
        s = s.lower().strip()
        s = re.sub(r"[\W_]+", " ", s)
        return norm_ws(s)

    def key(p: dict) -> str:
        return canon(p.get("asset")) + "||" + canon(p.get("indication"))

    merged: Dict[str, dict] = {}
    for p in all_programs:
        if not p.get("asset"):
            continue
        k = key(p)
        if k not in merged:
            merged[k] = p
        else:
            # if one has phase and the other doesn't, keep phase
            if not merged[k].get("phase") and p.get("phase"):
                merged[k]["phase"] = p.get("phase")
            # keep earliest page reference if present
            if merged[k].get("source_page") and p.get("source_page"):
                merged[k]["source_page"] = min(merged[k]["source_page"], p["source_page"])

    return {
        "as_of_date": as_of,
        "programs": list(merged.values()),
    }


# =============================
# Text-only extraction fallback
# =============================
def llm_extract_pipeline_text(company_name: str, source_text: str, models: List[str]) -> dict:
    prompt = f"""
You are a pharmaceutical competitive intelligence analyst.

TASK:
From the raw text below for {company_name}, extract a development pipeline list.

OUTPUT JSON SCHEMA:
{{
  "as_of_date": string | null,
  "programs": [
    {{
      "asset": string,
      "indication": string | null,
      "phase": string | null,
      "trial_or_program": string | null,
      "partner": string | null,
      "notes": string | null
    }}
  ]
}}

Rules:
- Return ONLY valid JSON.
- Do not invent assets.
- If phase is not explicit in text, set phase to null.
- If no pipeline content found, return {{"as_of_date": null, "programs": []}}.

RAW TEXT:
{source_text}
""".strip()

    try:
        out = gemini_generate_text(models=models, prompt=prompt, max_attempts_per_model=2)
        parsed = _try_parse_json(out)
        if parsed is not None:
            parsed.setdefault("as_of_date", None)
            parsed.setdefault("programs", [])
            return parsed
        return {"as_of_date": None, "programs": [], "_llm_parse_error": "Bad JSON", "_llm_raw_output": out[:8000]}
    except Exception as e:
        return {"as_of_date": None, "programs": [], "_llm_error": str(e)}


def llm_write_executive_brief(company_name: str, diff_obj: dict, models: List[str]) -> dict:
    payload = {"company": company_name, "diff": diff_obj}
    prompt = f"""
You are a senior pharma competitive intelligence analyst.

Write an executive briefing for {company_name} based on pipeline changes.

Return ONLY valid JSON:
{{
  "headline": string,
  "executive_summary": string,
  "why_it_matters": [string],
  "watchlist": [string]
}}

Rules:
- If no changes, say explicitly that the source doc appears unchanged.
INPUT_JSON:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    try:
        out = gemini_generate_text(models=models, prompt=prompt, max_attempts_per_model=2)
        parsed = _try_parse_json(out)
        if parsed:
            parsed.setdefault("headline", "Pipeline monitoring update")
            parsed.setdefault("executive_summary", "")
            parsed.setdefault("why_it_matters", [])
            parsed.setdefault("watchlist", [])
            return parsed
    except Exception:
        pass

    return {
        "headline": "Pipeline monitoring update",
        "executive_summary": "Narrative generation was unavailable; refer to the change log and pipeline inventory.",
        "why_it_matters": [],
        "watchlist": [],
    }


# =============================
# Diff + reporting helpers
# =============================
def canon(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[\W_]+", " ", s)
    return norm_ws(s)


def diff_programs(prev: dict, curr: dict) -> dict:
    prev_programs = prev.get("programs", []) if prev else []
    curr_programs = curr.get("programs", []) if curr else []

    def key(p: dict) -> str:
        return canon(p.get("asset")) + " || " + canon(p.get("indication"))

    prev_map = {key(p): p for p in prev_programs if p.get("asset")}
    curr_map = {key(p): p for p in curr_programs if p.get("asset")}

    added_keys = sorted(set(curr_map.keys()) - set(prev_map.keys()))
    removed_keys = sorted(set(prev_map.keys()) - set(curr_map.keys()))
    common_keys = sorted(set(prev_map.keys()) & set(curr_map.keys()))

    phase_changes = []
    for k in common_keys:
        p0 = prev_map[k]
        p1 = curr_map[k]
        if canon(p0.get("phase")) != canon(p1.get("phase")):
            phase_changes.append({
                "asset": p1.get("asset"),
                "indication": p1.get("indication"),
                "from_phase": p0.get("phase"),
                "to_phase": p1.get("phase"),
                "source_page": p1.get("source_page"),
            })

    return {"added": [curr_map[k] for k in added_keys], "removed": [prev_map[k] for k in removed_keys], "phase_changes": phase_changes}


def phase_counts(programs: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in programs:
        ph = p.get("phase") or "Unknown"
        counts[ph] = counts.get(ph, 0) + 1
    return counts


def confidence_score(d: dict) -> float:
    score = 0.35
    if d.get("phase_changes"):
        score += 0.30
    if d.get("added") or d.get("removed"):
        score += 0.20
    return max(0.0, min(0.95, score))


def export_programs_csv(path: str, programs: List[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asset", "indication", "phase", "trial_or_program", "partner", "notes", "source_page"])
        for p in programs:
            w.writerow([
                p.get("asset") or "",
                p.get("indication") or "",
                p.get("phase") or "",
                p.get("trial_or_program") or "",
                p.get("partner") or "",
                p.get("notes") or "",
                p.get("source_page") or "",
            ])


# =============================
# Reports (Markdown + PDF)
# =============================
def render_markdown_report(
    company_name: str,
    run_date: str,
    source_url: str,
    source_pdf_path: str,
    source_sha256: str,
    snapshot: dict,
    prev_exists: bool,
    reused_snapshot: bool,
    recovered_baseline: bool,
    d: dict,
    brief: dict,
    programs_csv_path: str,
) -> str:
    programs = snapshot.get("programs", []) or []
    as_of = snapshot.get("as_of_date") or "Unknown"
    counts = phase_counts(programs)

    lines: List[str] = []
    lines.append(f"# Competitive Pipeline Report — {company_name}\n")
    lines.append("## Executive summary")
    lines.append(f"- Run date (UTC): {run_date}")
    lines.append(f"- Source URL: {source_url}")
    lines.append(f"- Stored source PDF: `{source_pdf_path}`")
    lines.append(f"- Source SHA256: `{source_sha256}`")
    lines.append(f"- Extracted 'as of' date: {as_of}")
    lines.append(f"- Programs extracted: {len(programs)}")
    if recovered_baseline:
        lines.append("- Note: latest.json was invalid; recovered most recent valid dated snapshot as baseline.")
    if reused_snapshot:
        lines.append("- Note: Source PDF unchanged; reused prior snapshot to avoid extraction variability.")
    lines.append(f"- Full inventory export: `{programs_csv_path}`\n")

    lines.append(f"**{brief.get('headline', 'Pipeline monitoring update')}**\n")
    if brief.get("executive_summary"):
        lines.append(brief["executive_summary"] + "\n")

    if brief.get("why_it_matters"):
        lines.append("### Why it matters")
        for b in brief["why_it_matters"]:
            lines.append(f"- {b}")
        lines.append("")

    if brief.get("watchlist"):
        lines.append("### Watchlist")
        for b in brief["watchlist"]:
            lines.append(f"- {b}")
        lines.append("")

    lines.append("## Pipeline snapshot (counts by phase)")
    total = 0
    for k, v in sorted(counts.items(), key=lambda x: x[0]):
        lines.append(f"- {k}: {v}")
        total += v
    lines.append(f"- Total: {total}\n")

    lines.append("## Change log (previous vs current)")
    lines.append(f"- Confidence score (0–1): {confidence_score(d):.2f}\n")

    def section(title: str, items: List[dict]):
        lines.append(f"### {title}")
        if not items:
            lines.append("_None detected._\n")
            return
        for it in items[:200]:
            asset = it.get("asset")
            ind = it.get("indication")
            phase = it.get("phase")
            page = it.get("source_page")
            bits = [f"**{asset}**"]
            if ind:
                bits.append(ind)
            if phase:
                bits.append(f"({phase})")
            if page:
                bits.append(f"[p.{page}]")
            lines.append("- " + " — ".join(bits))
        lines.append("")

    if not prev_exists:
        lines.append("_Baseline created. Changes will appear on the next run._\n")
    else:
        section("New / Added items", d.get("added", []))
        section("Removed / Discontinued items", d.get("removed", []))
        lines.append("### Phase changes")
        pcs = d.get("phase_changes", [])
        if not pcs:
            lines.append("_None detected._\n")
        else:
            for c in pcs[:200]:
                lines.append(f"- **{c.get('asset')}** — {c.get('indication') or ''} — {c.get('from_phase')} → {c.get('to_phase')} (p.{c.get('source_page')})")
            lines.append("")

    # ALWAYS include inventory (this solves your “8.1 is empty” complaint)
    lines.append("## Pipeline inventory (full list)")
    lines.append("| Asset | Indication | Phase | Source page |")
    lines.append("|---|---|---|---|")
    for p in sorted(programs, key=lambda x: (x.get("phase") or "ZZZ", x.get("asset") or "")):
        lines.append(f"| {p.get('asset','')} | {p.get('indication','') or ''} | {p.get('phase') or ''} | {p.get('source_page') or ''} |")
    lines.append("")

    lines.append("## Citations / provenance")
    lines.append(f"- Source: {source_url}")
    lines.append(f"- Stored PDF: `{source_pdf_path}`")
    lines.append(f"- SHA256: `{source_sha256}`\n")

    return "\n".join(lines)


def build_pdf_report(
    pdf_path: str,
    company_name: str,
    run_date: str,
    source_url: str,
    source_pdf_path: str,
    source_sha256: str,
    snapshot: dict,
    prev_exists: bool,
    reused_snapshot: bool,
    recovered_baseline: bool,
    d: dict,
    brief: dict,
    programs_csv_path: str,
) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], spaceAfter=12))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], spaceAfter=8))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], leading=12, fontSize=9, spaceAfter=6))

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=LETTER,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=f"{company_name} Pipeline Report",
        author="Competitor Pipeline Agent",
    )

    def P(txt: str, style_name: str = "Bodyx") -> Paragraph:
        return Paragraph(xml_escape(txt), styles[style_name])

    programs = snapshot.get("programs", []) or []
    counts = phase_counts(programs)
    as_of = snapshot.get("as_of_date") or "Unknown"

    story: List[Any] = []
    story.append(P(f"Competitive Pipeline Report — {company_name}", "H1x"))
    story.append(P(f"Run date (UTC): {run_date}", "Bodyx"))

    prov_rows = [
        ["Source URL", source_url],
        ["Stored PDF", source_pdf_path],
        ["SHA256", source_sha256],
        ["Extracted 'as of' date", as_of],
        ["Programs extracted", str(len(programs))],
        ["Recovered baseline", "Yes" if recovered_baseline else "No"],
        ["Doc unchanged / reused", "Yes" if reused_snapshot else "No"],
        ["Full inventory (CSV)", programs_csv_path],
    ]
    tbl = Table(prov_rows, colWidths=[1.6 * inch, 4.9 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(Spacer(1, 10))
    story.append(tbl)
    story.append(Spacer(1, 14))

    story.append(P("Executive summary", "H2x"))
    story.append(P(brief.get("headline", "Pipeline monitoring update"), "Bodyx"))
    story.append(P(brief.get("executive_summary", ""), "Bodyx"))

    if recovered_baseline:
        story.append(P("Note: latest.json baseline was invalid; recovered most recent valid dated snapshot.", "Smallx"))
    if reused_snapshot:
        story.append(P("Note: Source PDF unchanged; reused prior snapshot to avoid extraction variability.", "Smallx"))

    # Snapshot table
    story.append(P("Pipeline snapshot (counts by phase)", "H2x"))
    rows = [["Phase", "Count"]] + [[k, str(v)] for k, v in sorted(counts.items(), key=lambda x: x[0])]
    rows.append(["Total", str(sum(counts.values()))])
    t2 = Table(rows, colWidths=[4.5 * inch, 1.0 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))

    # Change log (only delta)
    story.append(P("Change log (previous vs current)", "H2x"))
    story.append(P(f"Confidence score (0–1): {confidence_score(d):.2f}", "Bodyx"))

    def change_table(title: str, items: List[dict], mode: str) -> None:
        story.append(P(title, "Bodyx"))
        if not items:
            story.append(P("None detected.", "Bodyx"))
            story.append(Spacer(1, 6))
            return

        items = items[:60]
        if mode == "phase":
            rows = [["Asset", "Indication", "Phase change", "Page"]]
            for it in items:
                rows.append([
                    it.get("asset") or "",
                    it.get("indication") or "",
                    f"{it.get('from_phase')} → {it.get('to_phase')}",
                    str(it.get("source_page") or ""),
                ])
            widths = [2.0 * inch, 2.9 * inch, 1.2 * inch, 0.4 * inch]
        else:
            rows = [["Asset", "Indication", "Phase", "Page"]]
            for it in items:
                rows.append([
                    it.get("asset") or "",
                    it.get("indication") or "",
                    it.get("phase") or "",
                    str(it.get("source_page") or ""),
                ])
            widths = [2.0 * inch, 2.9 * inch, 1.2 * inch, 0.4 * inch]

        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    if prev_exists:
        change_table("New / Added items", d.get("added", []), mode="std")
        change_table("Removed / Discontinued items", d.get("removed", []), mode="std")
        change_table("Phase changes", d.get("phase_changes", []), mode="phase")
    else:
        story.append(P("Baseline created. Changes will appear on the next run.", "Bodyx"))

    # Appendix inventory (THIS makes the “new report is empty” problem go away)
    story.append(PageBreak())
    story.append(P("Appendix — Full pipeline inventory", "H2x"))
    story.append(P("Each row includes the source page number for manual verification.", "Smallx"))

    inv_rows = [["Asset", "Indication", "Phase", "Page"]]
    for p in sorted(programs, key=lambda x: (x.get("phase") or "ZZZ", x.get("asset") or "")):
        inv_rows.append([
            p.get("asset") or "",
            p.get("indication") or "",
            p.get("phase") or "",
            str(p.get("source_page") or ""),
        ])

    inv = Table(inv_rows, colWidths=[2.2 * inch, 3.0 * inch, 1.0 * inch, 0.4 * inch], repeatRows=1)
    inv.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ]))
    story.append(inv)

    doc.build(story)


# =============================
# Main
# =============================
def main() -> None:
    with open("config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    llm_models = cfg.get("llm_models") or [cfg.get("llm_model", "gemini-2.5-flash")]

    company = cfg.get("company")
    if not company:
        comps = cfg.get("companies") or []
        if not comps:
            raise RuntimeError("config.yml must contain either 'company' or non-empty 'companies'.")
        company = comps[0]

    company_name = company.get("name", "Johnson & Johnson")
    slug = company.get("slug", "jnj")
    run_date = utc_today_str()

    sources_dir = os.path.join("sources", slug)
    raw_dir = os.path.join("raw", slug)
    snap_dir = os.path.join("snapshots", slug)
    rep_dir = os.path.join("reports", slug)
    for d in [sources_dir, raw_dir, snap_dir, rep_dir]:
        ensure_dir(d)

    prev_path = os.path.join(snap_dir, "latest.json")
    prev_snapshot = safe_json_load(prev_path)
    recovered_baseline = False

    if program_count(prev_snapshot) == 0:
        recovered, _ = find_latest_valid_snapshot(snap_dir)
        if recovered:
            prev_snapshot = recovered
            recovered_baseline = True

    prev_exists = prev_snapshot is not None and program_count(prev_snapshot) > 0

    # Source URL
    source_type = company.get("source_type") or "jnj_q4cdn_auto"
    if source_type == "jnj_q4cdn_auto":
        source_url = discover_latest_jnj_pipeline_pdf(
            q4cdn_base=company["q4cdn_base"],
            filename_prefix=company.get("filename_prefix", "JNJ-Pipeline"),
            lookback_quarters=int(company.get("lookback_quarters", 10)),
        )
    elif source_type == "direct_pdf":
        source_url = company["pdf_url"]
    else:
        raise RuntimeError("Unsupported source_type. Use jnj_q4cdn_auto or direct_pdf.")

    # Download PDF
    pdf_resp = http_get(source_url)
    pdf_resp.raise_for_status()
    pdf_bytes = pdf_resp.content
    pdf_hash = sha256_bytes(pdf_bytes)

    # Store exact PDF used
    source_pdf_path = os.path.join(sources_dir, f"{run_date}.pdf")
    with open(source_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    safe_text_dump(
        os.path.join(raw_dir, f"{run_date}.source.txt"),
        f"Source URL: {source_url}\nStored PDF: {source_pdf_path}\nSHA256: {pdf_hash}\n",
    )

    # Reuse if unchanged
    prev_pdf_hash = best_prev_pdf_hash(prev_snapshot, sources_dir=sources_dir, snap_dir=snap_dir)
    reused_snapshot = False

    if prev_exists and prev_pdf_hash and prev_pdf_hash == pdf_hash:
        reused_snapshot = True
        snapshot = deep_copy(prev_snapshot)
        snapshot.setdefault("_meta", {})
        snapshot["_meta"].update({
            "run_date_utc": run_date,
            "source_url": source_url,
            "stored_pdf_path": source_pdf_path,
            "source_sha256": pdf_hash,
            "doc_unchanged": True,
            "extraction_ok": True,
        })
        d = {"added": [], "removed": [], "phase_changes": []}
        brief = {
            "headline": "No pipeline update detected",
            "executive_summary": (
                "The source pipeline PDF is unchanged since the last valid snapshot (same SHA256). "
                "This report reuses the prior extracted snapshot to avoid false differences caused by extraction variability."
            ),
            "why_it_matters": [],
            "watchlist": [],
        }
        safe_text_dump(os.path.join(raw_dir, f"{run_date}.extracted_text_preview.txt"),
                       "Reused previous snapshot (PDF unchanged). No new extraction performed.\n")
    else:
        # Extract (vision first, text/OCR fallback)
        use_vision = bool(company.get("use_vision", True))
        vision_max_pages = int(company.get("vision_max_pages", 8))
        vision_dpi = int(company.get("vision_dpi", 170))

        snapshot = {"as_of_date": None, "programs": []}
        extraction_mode = "none"
        errors: List[str] = []

        if use_vision:
            try:
                extraction_mode = "vision"
                snapshot = llm_extract_pipeline_vision(company_name, pdf_bytes, llm_models, max_pages=vision_max_pages, dpi=vision_dpi)
            except Exception as e:
                errors.append(f"vision_error: {e}")
                snapshot = {"as_of_date": None, "programs": []}

        if program_count(snapshot) == 0:
            try:
                extraction_mode = "text+ocr"
                extracted_text = pdf_to_text(pdf_bytes)
                combined_text = maybe_add_ocr(pdf_bytes, extracted_text, ocr_max_pages=8, ocr_dpi=220)
                safe_text_dump(os.path.join(raw_dir, f"{run_date}.extracted_text_preview.txt"), combined_text[:25000])
                snapshot = llm_extract_pipeline_text(company_name, combined_text[:160000], llm_models)
            except Exception as e:
                errors.append(f"text_ocr_error: {e}")

        snapshot.setdefault("_meta", {})
        snapshot["_meta"].update({
            "run_date_utc": run_date,
            "source_url": source_url,
            "stored_pdf_path": source_pdf_path,
            "source_sha256": pdf_hash,
            "extraction_mode": extraction_mode,
            "errors": errors,
        })

        pc = program_count(snapshot)
        snapshot["_meta"]["extraction_ok"] = pc > 0
        if pc == 0:
            snapshot["_meta"]["extraction_note"] = "0 programs extracted (vision and fallback failed). Baseline not overwritten."

        d = diff_programs(prev_snapshot or {}, snapshot)
        if pc == 0:
            brief = {
                "headline": "Extraction issue detected",
                "executive_summary": (
                    "The source PDF was downloaded successfully, but extraction returned 0 programs. "
                    "This run is treated as unreliable and will not overwrite the last good baseline."
                ),
                "why_it_matters": [],
                "watchlist": [],
            }
        else:
            brief = llm_write_executive_brief(company_name, d, llm_models)

    # Save dated snapshot always
    dated_snapshot_path = os.path.join(snap_dir, f"{run_date}.json")
    safe_json_dump(dated_snapshot_path, snapshot)

    # Update latest only if safe
    if reused_snapshot:
        safe_json_dump(prev_path, snapshot)
    else:
        if bool((snapshot.get("_meta", {}) or {}).get("extraction_ok", False)):
            safe_json_dump(prev_path, snapshot)

    # Export CSV (always; uses reused snapshot programs when unchanged)
    programs_csv_path = os.path.join(rep_dir, f"{run_date}.programs.csv")
    export_programs_csv(programs_csv_path, snapshot.get("programs", []) or [])

    # Reports
    md_path = os.path.join(rep_dir, f"{run_date}.md")
    pdf_report_path = os.path.join(rep_dir, f"{run_date}.pdf")
    changes_json_path = os.path.join(rep_dir, f"{run_date}.changes.json")

    md = render_markdown_report(
        company_name=company_name,
        run_date=run_date,
        source_url=source_url,
        source_pdf_path=source_pdf_path,
        source_sha256=pdf_hash,
        snapshot=snapshot,
        prev_exists=prev_exists,
        reused_snapshot=reused_snapshot,
        recovered_baseline=recovered_baseline,
        d=d,
        brief=brief,
        programs_csv_path=programs_csv_path,
    )
    safe_text_dump(md_path, md)

    build_pdf_report(
        pdf_path=pdf_report_path,
        company_name=company_name,
        run_date=run_date,
        source_url=source_url,
        source_pdf_path=source_pdf_path,
        source_sha256=pdf_hash,
        snapshot=snapshot,
        prev_exists=prev_exists,
        reused_snapshot=reused_snapshot,
        recovered_baseline=recovered_baseline,
        d=d,
        brief=brief,
        programs_csv_path=programs_csv_path,
    )

    safe_json_dump(changes_json_path, {
        "meta": snapshot.get("_meta", {}),
        "diff": d,
        "brief": brief,
        "reused_snapshot": reused_snapshot,
        "recovered_baseline": recovered_baseline,
    })

    print("Done.")
    print(f"Source URL: {source_url}")
    print(f"Stored source PDF: {source_pdf_path}")
    print(f"Snapshot: {dated_snapshot_path}")
    print(f"Report (MD): {md_path}")
    print(f"Report (PDF): {pdf_report_path}")
    print(f"Programs CSV: {programs_csv_path}")


if __name__ == "__main__":
    main()
