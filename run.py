import os
import re
import json
import time
import hashlib
import datetime
from typing import Optional, List, Tuple, Dict, Any

import yaml
import requests
from bs4 import BeautifulSoup

import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

from google import genai


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
    resp = requests.get(url, headers=headers or {}, timeout=timeout)
    return resp


# =============================
# PDF / HTML extraction
# =============================
def pdf_to_text(pdf_bytes: bytes) -> str:
    """Text extraction only (no OCR)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for i in range(doc.page_count):
        parts.append(doc.load_page(i).get_text("text"))
    return "\n".join(parts)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return norm_ws(soup.get_text(" "))


def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Light preprocessing improves OCR on slide/table screenshots:
    - convert to grayscale
    - increase contrast
    - optional autocontrast
    """
    img = pil_img.convert("L")
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    return img


def ocr_pdf_pages(pdf_bytes: bytes, max_pages: int = 20, dpi: int = 220) -> str:
    """
    OCR first N pages of a PDF using Tesseract.
    This is the critical fix for image/table-heavy pipeline PDFs.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = min(doc.page_count, max_pages)

    texts = []
    for i in range(n_pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi)
        mode = "RGB" if pix.alpha == 0 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

        img = preprocess_for_ocr(img)

        # Tesseract config: psm 6 often works OK for dense text blocks/tables
        t = pytesseract.image_to_string(img, config="--psm 6")
        t = t.strip()
        if t:
            texts.append(f"\n\n===== OCR PAGE {i+1} =====\n{t}")

    return "\n".join(texts)


def maybe_add_ocr(pdf_bytes: bytes, text_extracted: str, ocr_max_pages: int, ocr_dpi: int) -> str:
    """
    If text extraction is too sparse, add OCR text.
    """
    # Heuristic: if extracted text is very short, OCR is needed.
    # J&J pipeline PDFs often fall into this category.
    if len(text_extracted.strip()) >= 5000:
        return text_extracted

    ocr_text = ocr_pdf_pages(pdf_bytes, max_pages=ocr_max_pages, dpi=ocr_dpi)
    combined = (text_extracted or "") + "\n\n" + (ocr_text or "")
    return combined


# =============================
# SEC helpers (BioNTech)
# =============================
def sec_user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("Missing SEC_USER_AGENT env var (set it in GitHub Secrets).")
    return ua


def sec_headers_json() -> dict:
    return {
        "User-Agent": sec_user_agent(),
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }


def sec_headers_html() -> dict:
    return {
        "User-Agent": sec_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
    }


def sec_submissions_url(cik_10: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik_10}.json"


def accession_index_url(cik_no_leading_zeros: str, accession_no: str) -> str:
    acc_nodash = accession_no.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_leading_zeros}/{acc_nodash}/{accession_no}-index.html"


def find_latest_presentation_from_sec(cik_10: str, max_filings_to_check: int = 80) -> str:
    """
    Heuristic:
    - Fetch recent filings list from submissions JSON
    - Look at recent 6-K filings
    - Open the filing index page and search for exhibit filenames containing 'presentation' / 'deck' / 'slides'
    - Fallback to primary doc if it looks like an exhibit
    """
    cik_int = str(int(cik_10))  # remove leading zeros for Archives path
    sub_url = sec_submissions_url(cik_10)

    r = http_get(sub_url, headers=sec_headers_json())
    r.raise_for_status()
    data = r.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    n = min(len(forms), max_filings_to_check)

    for i in range(n):
        form = forms[i]
        accession = accession_numbers[i]
        primary = primary_docs[i] if i < len(primary_docs) else ""

        if form != "6-K":
            continue

        idx_url = accession_index_url(cik_int, accession)

        try:
            idx = http_get(idx_url, headers=sec_headers_html())
            if idx.status_code != 200:
                continue

            soup = BeautifulSoup(idx.text, "html.parser")
            candidates: List[str] = []

            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if re.search(r"(presentation|deck|slides)", href, re.IGNORECASE):
                    if href.startswith("http"):
                        candidates.append(href)
                    else:
                        candidates.append("https://www.sec.gov" + href)

            if candidates:
                return candidates[0]

            # Fallback: primary doc if it looks exhibit-like
            acc_nodash = accession.replace("-", "")
            primary_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary}"
            if re.search(r"(ex99|presentation|deck|slides|investor)", primary, re.IGNORECASE):
                return primary_url

        finally:
            # Be gentle with SEC.
            time.sleep(0.25)

    raise RuntimeError("Could not locate a recent presentation/deck from SEC filings for this CIK.")


# =============================
# Gemini LLM extraction
# =============================
def gemini_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY env var (set it in GitHub Secrets).")
    return genai.Client(api_key=key)


def _try_parse_json(text: str) -> Optional[dict]:
    """
    Robust JSON parse:
    - direct json.loads
    - if that fails, try to extract the first {...} block and parse that
    """
    text = (text or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt to extract a JSON object from within extra text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None


def llm_extract_pipeline(company_name: str, source_text: str, model: str) -> dict:
    """
    Converts extracted text (plus OCR if needed) into structured JSON pipeline.
    """
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
- Return ONLY valid JSON (no markdown, no commentary).
- If the text is not a pipeline, return: {{"as_of_date": null, "programs": []}}
- Do not invent assets. Only extract what is present.
- Phase should be one of: "Preclinical", "Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Registration", "Approved".
- If you see "Discovery", "Research", "IND-enabling", map to "Preclinical".
- If unsure about a field, set it to null.

RAW TEXT:
{source_text}
""".strip()

    client = gemini_client()
    resp = client.models.generate_content(model=model, contents=prompt)
    out = (resp.text or "").strip()

    parsed = _try_parse_json(out)
    if parsed is not None:
        # Ensure keys exist
        parsed.setdefault("as_of_date", None)
        parsed.setdefault("programs", [])
        return parsed

    # If parsing fails, return debug info instead of crashing
    return {
        "as_of_date": None,
        "programs": [],
        "_llm_parse_error": "Could not parse JSON from model output",
        "_llm_raw_output": out[:8000],
    }


# =============================
# Diff logic + report
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


def confidence_score(d: dict) -> float:
    score = 0.35
    if d.get("phase_changes"):
        score += 0.35
    if d.get("added") or d.get("removed"):
        score += 0.20
    return max(0.0, min(0.95, score))


def render_report(company: dict, source_url: str, prev_snapshot_path: str, curr_snapshot: dict, d: dict) -> str:
    name = company["name"]
    as_of = curr_snapshot.get("as_of_date")
    score = confidence_score(d)

    lines = []
    lines.append(f"# Pipeline Change Report — {name}")
    lines.append("")
    lines.append(f"- Run date (UTC): {utc_today_str()}")
    lines.append(f"- Source: {source_url}")
    lines.append(f"- Extracted 'as of' date: {as_of if as_of else 'Unknown'}")
    lines.append(f"- Confidence score (0–1): {score:.2f}")
    lines.append("")

    # Helpful debug for beginners: show if extraction looks empty
    prog_count = len(curr_snapshot.get("programs", []))
    lines.append(f"- Extracted program count: {prog_count}")
    if curr_snapshot.get("_llm_parse_error"):
        lines.append(f"- Extraction warning: {curr_snapshot.get('_llm_parse_error')}")
    lines.append("")

    def section(title: str, items: List[dict]):
        lines.append(f"## {title}")
        if not items:
            lines.append("_None detected._")
            lines.append("")
            return
        for it in items:
            asset = it.get("asset")
            ind = it.get("indication")
            phase = it.get("phase")
            trial = it.get("trial_or_program")
            partner = it.get("partner")
            bits = [f"**{asset}**"]
            if ind:
                bits.append(ind)
            if phase:
                bits.append(f"({phase})")
            if trial:
                bits.append(f"Trial/Program: {trial}")
            if partner:
                bits.append(f"Partner: {partner}")
            lines.append("- " + " — ".join(bits))
        lines.append("")

    section("New / Added items", d.get("added", []))
    section("Removed / Discontinued items", d.get("removed", []))

    lines.append("## Phase changes")
    pcs = d.get("phase_changes", [])
    if not pcs:
        lines.append("_None detected._")
        lines.append("")
    else:
        for c in pcs:
            lines.append(
                f"- **{c.get('asset')}** — {c.get('indication') or ''} — {c.get('from_phase')} → {c.get('to_phase')}"
            )
        lines.append("")

    lines.append("## Citations")
    lines.append(f"- {source_url}")
    lines.append(f"- Previous snapshot file: {prev_snapshot_path if os.path.exists(prev_snapshot_path) else 'None (first run)'}")
    lines.append("")

    return "\n".join(lines)


# =============================
# Company runner
# =============================
def run_company(company: dict, llm_model: str) -> Tuple[str, dict, dict, str]:
    slug = company["slug"]
    typ = company["type"]
    today = utc_today_str()

    raw_dir = os.path.join("raw", slug)
    snap_dir = os.path.join("snapshots", slug)
    rep_dir = os.path.join("reports", slug)

    ensure_dir(raw_dir)
    ensure_dir(snap_dir)
    ensure_dir(rep_dir)

    prev_path = os.path.join(snap_dir, "latest.json")
    prev = safe_json_load(prev_path)

    source_url = ""
    raw_text = ""

    # OCR controls (optional in config; safe defaults)
    ocr_max_pages = int(company.get("ocr_max_pages", 20))
    ocr_dpi = int(company.get("ocr_dpi", 220))

    if typ == "direct_pdf":
        pdf_url = company["pdf_url"]
        source_url = pdf_url

        pdf_resp = http_get(pdf_url)
        pdf_resp.raise_for_status()
        pdf_bytes = pdf_resp.content

        # text + OCR fallback
        extracted_text = pdf_to_text(pdf_bytes)
        combined_text = maybe_add_ocr(pdf_bytes, extracted_text, ocr_max_pages, ocr_dpi)
        raw_text = combined_text

        safe_text_dump(
            os.path.join(raw_dir, f"{today}.source.txt"),
            f"PDF URL: {pdf_url}\nSHA256: {sha256_bytes(pdf_bytes)}\n"
            f"Text chars: {len(extracted_text)} | Combined chars (with OCR if used): {len(combined_text)}\n",
        )

    elif typ == "sec_latest_presentation":
        cik = company["cik"]
        pres_url = find_latest_presentation_from_sec(cik)
        source_url = pres_url

        r = http_get(pres_url, headers=sec_headers_html())
        r.raise_for_status()

        if pres_url.lower().endswith(".pdf"):
            pdf_bytes = r.content
            extracted_text = pdf_to_text(pdf_bytes)
            combined_text = maybe_add_ocr(pdf_bytes, extracted_text, ocr_max_pages, ocr_dpi)
            raw_text = combined_text

            safe_text_dump(
                os.path.join(raw_dir, f"{today}.source.txt"),
                f"SEC URL (PDF): {pres_url}\nSHA256: {sha256_bytes(pdf_bytes)}\n"
                f"Text chars: {len(extracted_text)} | Combined chars (with OCR if used): {len(combined_text)}\n",
            )
        else:
            raw_text = html_to_text(r.text)
            safe_text_dump(os.path.join(raw_dir, f"{today}.source.txt"), f"SEC URL (HTML): {pres_url}\n")

    else:
        raise RuntimeError(f"Unknown company type: {typ}")

    # Save raw text preview for debugging (very helpful)
    safe_text_dump(os.path.join(raw_dir, f"{today}.extracted_text_preview.txt"), raw_text[:20000])

    # LLM extraction (cap input)
    input_cap = 160000
    extracted = llm_extract_pipeline(company["name"], raw_text[:input_cap], llm_model)

    extracted["_meta"] = {
        "run_date_utc": today,
        "source_url": source_url,
        "input_text_sha256": sha256_bytes(raw_text.encode("utf-8", errors="ignore")),
        "ocr_max_pages": ocr_max_pages,
        "ocr_dpi": ocr_dpi,
    }

    # Save snapshot
    dated_path = os.path.join(snap_dir, f"{today}.json")
    safe_json_dump(dated_path, extracted)
    safe_json_dump(prev_path, extracted)

    # If model output was not parseable JSON, save it for inspection
    if extracted.get("_llm_raw_output"):
        safe_text_dump(os.path.join(raw_dir, f"{today}.llm_raw_output.txt"), extracted["_llm_raw_output"])

    # Diff + report
    d = diff_programs(prev or {}, extracted)
    report_md = render_report(company, source_url, prev_path, extracted, d)
    report_path = os.path.join(rep_dir, f"{today}.md")
    safe_text_dump(report_path, report_md)

    return report_path, extracted, d, source_url


def main() -> None:
    with open("config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    llm_model = cfg.get("llm_model", "gemini-2.5-flash")
    companies = cfg.get("companies", [])

    print(f"Running pipeline agent for {len(companies)} companies using model={llm_model}")

    for c in companies:
        print(f"\n--- {c['name']} ---")
        report_path, snapshot, d, url = run_company(c, llm_model)
        print(f"Source: {url}")
        print(f"Report written: {report_path}")
        print(f"Extracted programs: {len(snapshot.get('programs', []))}")
        print(f"Added: {len(d.get('added', []))} | Removed: {len(d.get('removed', []))} | Phase changes: {len(d.get('phase_changes', []))}")


if __name__ == "__main__":
    main()
