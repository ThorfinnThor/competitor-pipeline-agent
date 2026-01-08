import os
import re
import json
import time
import hashlib
import datetime
from typing import Optional, List, Dict, Any, Tuple

import yaml
import requests

import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

from google import genai

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from xml.sax.saxutils import escape as xml_escape


# =============================
# Utilities
# =============================
def utc_today_str() -> str:
    return datetime.datetime.utcnow().date().isoformat()


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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
    """
    Pattern:
      {base}/{YYYY}/q{q}/{prefix}-{q}Q{YY}.pdf
    Example:
      .../2025/q3/JNJ-Pipeline-3Q25.pdf
    """
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

    # Fallback: attempt GETs
    for url in candidates:
        try:
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code == 200:
                return url
        except Exception:
            pass

    raise RuntimeError("Could not auto-discover a valid J&J pipeline PDF in the lookback window.")


# =============================
# PDF extraction + OCR
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


def ocr_pdf_pages(pdf_bytes: bytes, max_pages: int = 12, dpi: int = 220) -> str:
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
    # If text extraction is sparse, add OCR
    if len((text_extracted or "").strip()) >= 5000:
        return text_extracted
    return (text_extracted or "") + "\n\n" + ocr_pdf_pages(pdf_bytes, max_pages=ocr_max_pages, dpi=ocr_dpi)


# =============================
# Gemini helpers (fallback + robust JSON)
# =============================
def gemini_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY env var (set it in GitHub Secrets).")
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


def gemini_generate_with_fallback(models: List[str], prompt: str, max_attempts_per_model: int = 2) -> str:
    """
    Tries models in order. Retries transient failures briefly.
    """
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

    raise RuntimeError(f"All Gemini model attempts failed. Last error: {last_err}")


def llm_extract_pipeline(company_name: str, source_text: str, models: List[str]) -> dict:
    prompt = f"""
You are a pharmaceutical competitive intelligence analyst.

TASK:
From the raw text below for {company_name}, extract a development pipeline list.

OUTPUT JSON SCHEMA (must follow exactly):
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

RULES:
- Return ONLY valid JSON (no markdown).
- Do not invent assets. Only extract what is present.
- Phase should be one of: "Preclinical", "Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Registration", "Approved".
- If unsure about a field, set it to null.
- If no pipeline content is found, return {{"as_of_date": null, "programs": []}}.

RAW TEXT:
{source_text}
""".strip()

    try:
        out = gemini_generate_with_fallback(models=models, prompt=prompt, max_attempts_per_model=2)
        parsed = _try_parse_json(out)
        if parsed is not None:
            parsed.setdefault("as_of_date", None)
            parsed.setdefault("programs", [])
            return parsed
        return {
            "as_of_date": None,
            "programs": [],
            "_llm_parse_error": "Could not parse JSON from model output",
            "_llm_raw_output": out[:8000],
        }
    except Exception as e:
        # Do not crash the run; degrade gracefully
        return {
            "as_of_date": None,
            "programs": [],
            "_llm_error": str(e),
        }


def llm_write_executive_brief(company_name: str, diff_obj: dict, models: List[str]) -> dict:
    """
    Optional narrative. If LLM fails, return a deterministic minimal brief.
    """
    payload = {"company": company_name, "diff": diff_obj}
    prompt = f"""
You are a senior pharma competitive intelligence analyst.

Write an executive briefing for {company_name} based on the pipeline changes.

Return ONLY valid JSON:
{{
  "headline": string,
  "executive_summary": string,
  "why_it_matters": [string],
  "watchlist": [string]
}}

Rules:
- executive_summary: 4–7 sentences.
- why_it_matters: 3–6 bullets, decision-oriented.
- watchlist: 3–6 bullets.
- If there are no changes, explain that the document may be unchanged or this run is a baseline.

INPUT_JSON:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    try:
        out = gemini_generate_with_fallback(models=models, prompt=prompt, max_attempts_per_model=2)
        parsed = _try_parse_json(out)
        if parsed:
            parsed.setdefault("headline", "Pipeline monitoring update")
            parsed.setdefault("executive_summary", "")
            parsed.setdefault("why_it_matters", [])
            parsed.setdefault("watchlist", [])
            return parsed
        raise RuntimeError("Narrative JSON parse failed.")
    except Exception:
        # deterministic fallback
        return {
            "headline": "Pipeline monitoring update",
            "executive_summary": (
                "This monitoring run completed successfully. "
                "Narrative generation was unavailable; please refer to the change log and provenance section."
            ),
            "why_it_matters": [],
            "watchlist": [],
        }


# =============================
# Diff logic
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
            })

    return {
        "added": [curr_map[k] for k in added_keys],
        "removed": [prev_map[k] for k in removed_keys],
        "phase_changes": phase_changes,
    }


def phase_counts(programs: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in programs:
        ph = p.get("phase") or "Unknown"
        counts[ph] = counts.get(ph, 0) + 1
    return counts


def confidence_score(d: dict) -> float:
    # Conservative until you add trial-registry corroboration
    score = 0.35
    if d.get("phase_changes"):
        score += 0.30
    if d.get("added") or d.get("removed"):
        score += 0.20
    return max(0.0, min(0.95, score))


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
    d: dict,
    brief: dict,
) -> str:
    programs = snapshot.get("programs", []) or []
    as_of = snapshot.get("as_of_date") or "Unknown"
    conf = confidence_score(d)
    counts = phase_counts(programs)

    meta = snapshot.get("_meta", {}) or {}
    extraction_ok = meta.get("extraction_ok", True)

    lines: List[str] = []
    lines.append(f"# Competitive Pipeline Report — {company_name}")
    lines.append("")
    lines.append("## Executive summary")
    lines.append(f"- Run date (UTC): {run_date}")
    lines.append(f"- Source URL: {source_url}")
    lines.append(f"- Stored source PDF: `{source_pdf_path}`")
    lines.append(f"- Source SHA256: `{source_sha256}`")
    lines.append(f"- Extracted 'as of' date: {as_of}")
    lines.append(f"- Programs extracted: {len(programs)}")
    lines.append(f"- Extraction OK: {extraction_ok}")
    if reused_snapshot:
        lines.append(f"- Note: PDF unchanged; reused prior snapshot to avoid extraction variability.")
    lines.append("")

    lines.append(f"**{brief.get('headline', '')}**")
    lines.append("")
    if brief.get("executive_summary"):
        lines.append(brief["executive_summary"])
        lines.append("")

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
    if counts:
        for k, v in sorted(counts.items(), key=lambda x: x[0]):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("_No programs extracted._")
    lines.append("")

    lines.append("## Change log (previous vs current)")
    lines.append(f"- Confidence score (0–1): {conf:.2f}")
    lines.append("")
    if not prev_exists:
        lines.append("_Baseline created. Changes will appear on the next run._")
        lines.append("")
    else:
        def section(title: str, items: List[dict]):
            lines.append(f"### {title}")
            if not items:
                lines.append("_None detected._")
                lines.append("")
                return
            for it in items[:200]:
                asset = it.get("asset")
                ind = it.get("indication")
                phase = it.get("phase")
                bits = [f"**{asset}**"]
                if ind:
                    bits.append(ind)
                if phase:
                    bits.append(f"({phase})")
                lines.append("- " + " — ".join(bits))
            if len(items) > 200:
                lines.append(f"- … truncated ({len(items)} total)")
            lines.append("")

        section("New / Added items", d.get("added", []))
        section("Removed / Discontinued items", d.get("removed", []))

        lines.append("### Phase changes")
        pcs = d.get("phase_changes", [])
        if not pcs:
            lines.append("_None detected._\n")
        else:
            for c in pcs[:200]:
                lines.append(f"- **{c.get('asset')}** — {c.get('indication') or ''} — {c.get('from_phase')} → {c.get('to_phase')}")
            if len(pcs) > 200:
                lines.append(f"- … truncated ({len(pcs)} total)")
            lines.append("")

    lines.append("## Citations / provenance")
    lines.append(f"- Source: {source_url}")
    lines.append(f"- Stored PDF: `{source_pdf_path}`")
    lines.append(f"- SHA256: `{source_sha256}`")
    lines.append("")

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
    d: dict,
    brief: dict,
) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], spaceAfter=12))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], spaceAfter=8))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], leading=14, spaceAfter=8))

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

    story: List[Any] = []
    story.append(P(f"Competitive Pipeline Report — {company_name}", "H1x"))
    story.append(P(f"Run date (UTC): {run_date}", "Bodyx"))

    meta = snapshot.get("_meta", {}) or {}
    extraction_ok = bool(meta.get("extraction_ok", True))

    prov_rows = [
        ["Source URL", source_url],
        ["Stored PDF", source_pdf_path],
        ["SHA256", source_sha256],
        ["Extracted 'as of' date", snapshot.get("as_of_date") or "Unknown"],
        ["Programs extracted", str(len(snapshot.get("programs", []) or []))],
        ["Extraction OK", str(extraction_ok)],
        ["Doc unchanged / reused", "Yes" if reused_snapshot else "No"],
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

    if reused_snapshot:
        story.append(P("Note: Source PDF is unchanged; the prior extracted snapshot was reused to avoid extraction variability.", "Bodyx"))

    if brief.get("why_it_matters"):
        story.append(P("Why it matters", "H2x"))
        bullets = "<br/>".join([f"• {xml_escape(x)}" for x in brief["why_it_matters"]])
        story.append(Paragraph(bullets, styles["Bodyx"]))

    if brief.get("watchlist"):
        story.append(P("Watchlist", "H2x"))
        bullets = "<br/>".join([f"• {xml_escape(x)}" for x in brief["watchlist"]])
        story.append(Paragraph(bullets, styles["Bodyx"]))

    # Snapshot counts
    story.append(P("Pipeline snapshot (counts by phase)", "H2x"))
    counts = phase_counts(snapshot.get("programs", []) or [])
    if counts:
        rows = [["Phase", "Count"]] + [[k, str(v)] for k, v in sorted(counts.items(), key=lambda x: x[0])]
        t2 = Table(rows, colWidths=[4.5 * inch, 1.0 * inch])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ]))
        story.append(t2)
    else:
        story.append(P("No programs extracted in this run.", "Bodyx"))
    story.append(Spacer(1, 12))

    # Change log
    story.append(P("Change log (previous vs current)", "H2x"))
    story.append(P(f"Confidence score (0–1): {confidence_score(d):.2f}", "Bodyx"))

    if not prev_exists:
        story.append(P("Baseline created. Changes will appear on the next run.", "Bodyx"))
    else:
        def change_table(title: str, items: List[dict], mode: str) -> None:
            story.append(P(title, "Bodyx"))
            if not items:
                story.append(P("None detected.", "Bodyx"))
                story.append(Spacer(1, 6))
                return

            items = items[:60]  # keep PDF readable
            if mode == "phase":
                rows = [["Asset", "Indication", "Phase change"]]
                for it in items:
                    rows.append([
                        it.get("asset") or "",
                        it.get("indication") or "",
                        f"{it.get('from_phase')} → {it.get('to_phase')}",
                    ])
                widths = [2.2 * inch, 3.0 * inch, 1.3 * inch]
            else:
                rows = [["Asset", "Indication", "Phase"]]
                for it in items:
                    rows.append([
                        it.get("asset") or "",
                        it.get("indication") or "",
                        it.get("phase") or "",
                    ])
                widths = [2.2 * inch, 3.0 * inch, 1.3 * inch]

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

        change_table("New / Added items", d.get("added", []), mode="std")
        change_table("Removed / Discontinued items", d.get("removed", []), mode="std")
        change_table("Phase changes", d.get("phase_changes", []), mode="phase")

    doc.build(story)


# =============================
# Main
# =============================
def main() -> None:
    with open("config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Models config: prefer llm_models list, fallback to llm_model
    llm_models = cfg.get("llm_models") or [cfg.get("llm_model", "gemini-2.5-flash")]

    company = cfg.get("company") or {}
    company_name = company.get("name", "Johnson & Johnson")
    slug = company.get("slug", "jnj")

    run_date = utc_today_str()

    # Dirs
    sources_dir = os.path.join("sources", slug)
    raw_dir = os.path.join("raw", slug)
    snap_dir = os.path.join("snapshots", slug)
    rep_dir = os.path.join("reports", slug)
    for d in [sources_dir, raw_dir, snap_dir, rep_dir]:
        ensure_dir(d)

    # Load previous baseline
    prev_path = os.path.join(snap_dir, "latest.json")
    prev_snapshot = safe_json_load(prev_path)
    prev_exists = prev_snapshot is not None
    reused_snapshot = False

    # Discover source URL
    source_type = company.get("source_type", "jnj_q4cdn_auto")
    if source_type == "jnj_q4cdn_auto":
        source_url = discover_latest_jnj_pipeline_pdf(
            q4cdn_base=company["q4cdn_base"],
            filename_prefix=company.get("filename_prefix", "JNJ-Pipeline"),
            lookback_quarters=int(company.get("lookback_quarters", 10)),
        )
    elif source_type == "direct_pdf":
        source_url = company["pdf_url"]
    else:
        raise RuntimeError("Unsupported source_type for this J&J-only run.py. Use jnj_q4cdn_auto or direct_pdf.")

    # Download PDF
    pdf_resp = http_get(source_url)
    pdf_resp.raise_for_status()
    pdf_bytes = pdf_resp.content
    pdf_hash = sha256_bytes(pdf_bytes)

    # Store exact PDF used (audit trail)
    source_pdf_path = os.path.join(sources_dir, f"{run_date}.pdf")
    with open(source_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    safe_text_dump(
        os.path.join(raw_dir, f"{run_date}.source.txt"),
        f"Source URL: {source_url}\nStored PDF: {source_pdf_path}\nSHA256: {pdf_hash}\n",
    )

    # Reuse snapshot if document unchanged and previous snapshot is valid
    if prev_snapshot:
        prev_meta = prev_snapshot.get("_meta", {}) or {}
        prev_hash = prev_meta.get("source_sha256")
        prev_count = len(prev_snapshot.get("programs", []) or [])

        if prev_hash == pdf_hash and prev_count > 0:
            reused_snapshot = True
            snapshot = deep_copy(prev_snapshot)
            snapshot.setdefault("_meta", {})
            snapshot["_meta"].update({
                "run_date_utc": run_date,
                "source_url": source_url,
                "stored_pdf_path": source_pdf_path,
                "source_sha256": pdf_hash,
                "doc_unchanged": True,
                "doc_unchanged_note": "Source PDF SHA256 unchanged; reused prior snapshot to avoid extraction variability.",
                "extraction_ok": True,
            })

            d = {"added": [], "removed": [], "phase_changes": []}
            brief = {
                "headline": "No pipeline update detected",
                "executive_summary": (
                    "The source pipeline PDF is unchanged since the last run (same SHA256). "
                    "This report reuses the prior extracted snapshot to avoid false differences caused by OCR/LLM variability."
                ),
                "why_it_matters": [],
                "watchlist": [],
            }

            safe_text_dump(
                os.path.join(raw_dir, f"{run_date}.extracted_text_preview.txt"),
                "Reused previous snapshot (PDF unchanged). No new extraction performed.\n",
            )

    # If not reused, perform extraction
    if not reused_snapshot:
        extracted_text = pdf_to_text(pdf_bytes)
        combined_text = maybe_add_ocr(
            pdf_bytes,
            extracted_text,
            ocr_max_pages=int(company.get("ocr_max_pages", 12)),
            ocr_dpi=int(company.get("ocr_dpi", 220)),
        )
        safe_text_dump(os.path.join(raw_dir, f"{run_date}.extracted_text_preview.txt"), combined_text[:25000])

        snapshot = llm_extract_pipeline(company_name, combined_text[:160000], llm_models)
        snapshot.setdefault("_meta", {})
        snapshot["_meta"].update({
            "run_date_utc": run_date,
            "source_url": source_url,
            "stored_pdf_path": source_pdf_path,
            "source_sha256": pdf_hash,
            "input_text_sha256": sha256_bytes(combined_text.encode("utf-8", errors="ignore")),
            "doc_unchanged": False,
        })

        # Determine extraction health
        program_count = len(snapshot.get("programs", []) or [])
        snapshot["_meta"]["extraction_ok"] = program_count > 0
        if program_count == 0:
            snapshot["_meta"]["extraction_note"] = "0 programs extracted (likely OCR/LLM issue). Baseline not overwritten."

        # Diff and narrative
        d = diff_programs(prev_snapshot or {}, snapshot)
        if not prev_exists:
            brief = {
                "headline": "Baseline pipeline snapshot created",
                "executive_summary": (
                    "This is the first successful snapshot in this repository context. "
                    "Future runs will compare new source documents against this baseline to detect pipeline changes."
                ),
                "why_it_matters": [],
                "watchlist": [],
            }
        elif program_count == 0:
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

    # Update latest.json only if safe
    if reused_snapshot:
        safe_json_dump(prev_path, snapshot)
    else:
        ok = bool((snapshot.get("_meta", {}) or {}).get("extraction_ok", False))
        if ok:
            safe_json_dump(prev_path, snapshot)

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
        d=d,
        brief=brief,
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
        d=d,
        brief=brief,
    )

    safe_json_dump(changes_json_path, {
        "meta": snapshot.get("_meta", {}),
        "diff": d,
        "brief": brief,
        "reused_snapshot": reused_snapshot,
    })

    print("Done.")
    print(f"Source URL: {source_url}")
    print(f"Stored source PDF: {source_pdf_path}")
    print(f"Snapshot: {dated_snapshot_path}")
    print(f"Report (MD): {md_path}")
    print(f"Report (PDF): {pdf_report_path}")


if __name__ == "__main__":
    main()
