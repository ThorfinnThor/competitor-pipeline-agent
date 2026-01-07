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
    return requests.get(url, headers=headers or {}, timeout=timeout)


def http_head_ok(url: str, timeout: int = 30) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code == 200:
            ct = (r.headers.get("Content-Type") or "").lower()
            # Content-Type might be missing; accept 200 anyway
            return True if not ct else ("pdf" in ct or "octet-stream" in ct)
        return False
    except Exception:
        return False


# =============================
# PDF / HTML extraction + OCR
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
    img = pil_img.convert("L")
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    return img


def ocr_pdf_pages(pdf_bytes: bytes, max_pages: int = 12, dpi: int = 220) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = min(doc.page_count, max_pages)

    texts = []
    for i in range(n_pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi)
        mode = "RGB" if pix.alpha == 0 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        img = preprocess_for_ocr(img)

        t = pytesseract.image_to_string(img, config="--psm 6")
        t = t.strip()
        if t:
            texts.append(f"\n\n===== OCR PAGE {i+1} =====\n{t}")

    return "\n".join(texts)


def maybe_add_ocr(pdf_bytes: bytes, text_extracted: str, ocr_max_pages: int, ocr_dpi: int) -> str:
    if len((text_extracted or "").strip()) >= 5000:
        return text_extracted
    ocr_text = ocr_pdf_pages(pdf_bytes, max_pages=ocr_max_pages, dpi=ocr_dpi)
    return (text_extracted or "") + "\n\n" + (ocr_text or "")


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
                    candidates.append(href if href.startswith("http") else "https://www.sec.gov" + href)

            if candidates:
                return candidates[0]

            acc_nodash = accession.replace("-", "")
            primary_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary}"
            if re.search(r"(ex99|presentation|deck|slides|investor)", primary, re.IGNORECASE):
                return primary_url

        finally:
            time.sleep(0.25)

    raise RuntimeError("Could not locate a recent presentation/deck from SEC filings for this CIK.")


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
    Try recent quarters first and pick the newest URL that exists.
    URL pattern observed in your working example:
      {base}/{YYYY}/q{q}/{prefix}-{q}Q{YY}.pdf
    Example:
      .../2025/q3/JNJ-Pipeline-3Q25.pdf
    """
    today = datetime.datetime.utcnow().date()
    y = today.year
    q = quarter_of_date(today)

    candidates = []
    for _ in range(max(1, lookback_quarters)):
        yy = y % 100
        url = f"{q4cdn_base}/{y}/q{q}/{filename_prefix}-{q}Q{yy:02d}.pdf"
        candidates.append((y, q, url))
        y, q = prev_quarter(y, q)

    # Check newest-to-oldest; first hit wins
    for y, q, url in candidates:
        if http_head_ok(url):
            return url

    # If HEAD is blocked (rare), try a tiny GET
    for y, q, url in candidates:
        try:
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code == 200:
                return url
        except Exception:
            pass

    raise RuntimeError("Could not auto-discover a valid J&J pipeline PDF in the lookback window.")


# =============================
# Gemini LLM extraction + “why it matters”
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


def llm_extract_pipeline(company_name: str, source_text: str, model: str) -> dict:
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
        parsed.setdefault("as_of_date", None)
        parsed.setdefault("programs", [])
        return parsed

    return {
        "as_of_date": None,
        "programs": [],
        "_llm_parse_error": "Could not parse JSON from model output",
        "_llm_raw_output": out[:8000],
    }


def llm_why_it_matters(company_name: str, changes: List[dict], model: str) -> List[dict]:
    """
    Given structured changes, return parallel list with why_it_matters + watch_items.
    """
    if not changes:
        return []

    payload = {"changes": changes}

    prompt = f"""
You are a senior pharma competitive intelligence analyst.

Given these pipeline changes for {company_name}, write decision-oriented implications.

Return ONLY valid JSON:
{{
  "analyses": [
    {{
      "change_id": string,
      "why_it_matters": string,
      "watch_items": [string]
    }}
  ]
}}

Rules:
- Be concrete and brief.
- Do not speculate wildly; if uncertain, say what additional evidence would raise confidence.
- Keep why_it_matters to 1–3 sentences.
- watch_items should be 1–4 short bullets.

CHANGES_JSON:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    client = gemini_client()
    resp = client.models.generate_content(model=model, contents=prompt)
    out = (resp.text or "").strip()

    parsed = _try_parse_json(out)
    if parsed and isinstance(parsed.get("analyses"), list):
        return parsed["analyses"]

    return []


# =============================
# Diff + classification
# =============================
def canon(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[\W_]+", " ", s)
    return norm_ws(s)


def phase_rank(phase: Optional[str]) -> int:
    order = {
        "Preclinical": 1,
        "Phase 1": 2,
        "Phase 1/2": 3,
        "Phase 2": 4,
        "Phase 2/3": 5,
        "Phase 3": 6,
        "Registration": 7,
        "Approved": 8,
    }
    return order.get(phase or "", 0)


def index_programs(programs: List[dict]) -> Dict[str, List[dict]]:
    by_asset = {}
    for p in programs:
        asset = p.get("asset")
        if not asset:
            continue
        k = canon(asset)
        by_asset.setdefault(k, []).append(p)
    return by_asset


def classify_changes(prev: dict, curr: dict) -> List[dict]:
    prev_programs = prev.get("programs", []) if prev else []
    curr_programs = curr.get("programs", []) if curr else []

    prev_by_asset = index_programs(prev_programs)
    curr_by_asset = index_programs(curr_programs)

    changes: List[dict] = []

    # New / removed assets
    prev_assets = set(prev_by_asset.keys())
    curr_assets = set(curr_by_asset.keys())

    for a in sorted(curr_assets - prev_assets):
        example = curr_by_asset[a][0]
        changes.append({
            "change_id": f"new_asset:{a}",
            "type": "NEW_ASSET",
            "asset": example.get("asset"),
            "details": {"added_rows": curr_by_asset[a]},
        })

    for a in sorted(prev_assets - curr_assets):
        example = prev_by_asset[a][0]
        changes.append({
            "change_id": f"removed_asset:{a}",
            "type": "REMOVED_ASSET",
            "asset": example.get("asset"),
            "details": {"removed_rows": prev_by_asset[a]},
        })

    # Asset present in both: compare indications + phase + partner
    for a in sorted(prev_assets & curr_assets):
        prev_rows = prev_by_asset[a]
        curr_rows = curr_by_asset[a]

        def row_key(r: dict) -> str:
            return canon(r.get("indication"))

        prev_map = {row_key(r): r for r in prev_rows}
        curr_map = {row_key(r): r for r in curr_rows}

        prev_inds = set(prev_map.keys())
        curr_inds = set(curr_map.keys())

        for ind in sorted(curr_inds - prev_inds):
            r = curr_map[ind]
            changes.append({
                "change_id": f"new_ind:{a}:{ind}",
                "type": "NEW_INDICATION",
                "asset": r.get("asset"),
                "indication": r.get("indication"),
                "details": {"current": r},
            })

        for ind in sorted(prev_inds - curr_inds):
            r = prev_map[ind]
            changes.append({
                "change_id": f"removed_ind:{a}:{ind}",
                "type": "REMOVED_INDICATION",
                "asset": r.get("asset"),
                "indication": r.get("indication"),
                "details": {"previous": r},
            })

        for ind in sorted(prev_inds & curr_inds):
            p0 = prev_map[ind]
            p1 = curr_map[ind]

            # Phase changes
            if canon(p0.get("phase")) != canon(p1.get("phase")):
                changes.append({
                    "change_id": f"phase:{a}:{ind}",
                    "type": "PHASE_CHANGE",
                    "asset": p1.get("asset"),
                    "indication": p1.get("indication"),
                    "details": {"from": p0.get("phase"), "to": p1.get("phase")},
                })

            # Partner changes
            if canon(p0.get("partner")) != canon(p1.get("partner")):
                changes.append({
                    "change_id": f"partner:{a}:{ind}",
                    "type": "PARTNERSHIP_CHANGE",
                    "asset": p1.get("asset"),
                    "indication": p1.get("indication"),
                    "details": {"from": p0.get("partner"), "to": p1.get("partner")},
                })

    return changes


def base_confidence_for_change(change_type: str) -> float:
    # Before registry corroboration, keep conservative.
    return {
        "NEW_ASSET": 0.65,
        "REMOVED_ASSET": 0.60,
        "NEW_INDICATION": 0.60,
        "REMOVED_INDICATION": 0.55,
        "PHASE_CHANGE": 0.75,
        "PARTNERSHIP_CHANGE": 0.70,
    }.get(change_type, 0.50)


def phase_counts(programs: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in programs:
        ph = p.get("phase") or "Unknown"
        counts[ph] = counts.get(ph, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: phase_rank(x[0])))


# =============================
# Reporting
# =============================
def render_report(
    company: dict,
    source_url: str,
    prev_snapshot_path: str,
    curr_snapshot: dict,
    changes: List[dict],
    analyses: List[dict],
) -> str:
    name = company["name"]
    as_of = curr_snapshot.get("as_of_date")
    programs = curr_snapshot.get("programs", []) or []
    counts = phase_counts(programs)

    analysis_by_id = {a.get("change_id"): a for a in analyses if a.get("change_id")}

    lines: List[str] = []
    lines.append(f"# Competitive Pipeline Report — {name}")
    lines.append("")
    lines.append("## Executive summary")
    lines.append(f"- Run date (UTC): {utc_today_str()}")
    lines.append(f"- Source: {source_url}")
    lines.append(f"- Extracted 'as of' date: {as_of if as_of else 'Unknown'}")
    lines.append(f"- Programs extracted: {len(programs)}")
    lines.append("")

    lines.append("### Pipeline size by phase (snapshot)")
    if counts:
        for k, v in counts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- No phase counts available (extraction likely empty).")
    lines.append("")

    # Change log
    lines.append("## Change log (previous vs current)")
    if not changes:
        lines.append("_No changes detected._")
        lines.append("")
        lines.append("Note: On the first successful run, this report establishes a baseline. Changes appear on the next run.")
        lines.append("")
    else:
        for ch in changes:
            ctype = ch.get("type")
            asset = ch.get("asset")
            ind = ch.get("indication")
            conf = base_confidence_for_change(ctype)
            what = ""

            if ctype == "PHASE_CHANGE":
                what = f"{asset} — {ind or ''}: phase {ch['details'].get('from')} → {ch['details'].get('to')}"
            elif ctype == "PARTNERSHIP_CHANGE":
                what = f"{asset} — {ind or ''}: partner {ch['details'].get('from')} → {ch['details'].get('to')}"
            elif ctype == "NEW_ASSET":
                what = f"{asset}: new asset added to pipeline"
            elif ctype == "REMOVED_ASSET":
                what = f"{asset}: removed from pipeline (possible discontinuation or deprioritization)"
            elif ctype == "NEW_INDICATION":
                what = f"{asset}: new indication added — {ind}"
            elif ctype == "REMOVED_INDICATION":
                what = f"{asset}: indication removed — {ind}"
            else:
                what = f"{asset}: change detected"

            lines.append(f"### {ctype}")
            lines.append(f"- What changed: {what}")
            lines.append(f"- Confidence (pre-corroboration): {conf:.2f}")

            a = analysis_by_id.get(ch.get("change_id"))
            if a:
                lines.append(f"- Why it matters: {a.get('why_it_matters')}")
                wis = a.get("watch_items") or []
                if wis:
                    lines.append("- Watch items:")
                    for w in wis:
                        lines.append(f"  - {w}")
            else:
                lines.append("- Why it matters: (not available)")

            lines.append(f"- Citations: {source_url}")
            lines.append("")

    # Appendix: small preview of pipeline rows
    lines.append("## Appendix: pipeline preview (first 15 rows)")
    if not programs:
        lines.append("_No programs extracted._")
    else:
        for p in programs[:15]:
            bits = [p.get("asset", "")]
            if p.get("indication"):
                bits.append(p["indication"])
            if p.get("phase"):
                bits.append(f"({p['phase']})")
            if p.get("partner"):
                bits.append(f"Partner: {p['partner']}")
            lines.append("- " + " — ".join([b for b in bits if b]))
    lines.append("")

    lines.append("## Citations and provenance")
    lines.append(f"- Source URL: {source_url}")
    lines.append(f"- Previous snapshot file: {prev_snapshot_path if os.path.exists(prev_snapshot_path) else 'None (baseline)'}")
    lines.append("")

    return "\n".join(lines)


# =============================
# Company runner
# =============================
def run_company(company: dict, llm_model: str) -> Tuple[str, dict, List[dict], str]:
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

    ocr_max_pages = int(company.get("ocr_max_pages", 12))
    ocr_dpi = int(company.get("ocr_dpi", 220))

    if typ == "jnj_q4cdn_auto":
        q4cdn_base = company["q4cdn_base"]
        prefix = company.get("filename_prefix", "JNJ-Pipeline")
        lookback = int(company.get("lookback_quarters", 8))

        pdf_url = discover_latest_jnj_pipeline_pdf(q4cdn_base, prefix, lookback)
        source_url = pdf_url

        pdf_resp = http_get(pdf_url)
        pdf_resp.raise_for_status()
        pdf_bytes = pdf_resp.content

        extracted_text = pdf_to_text(pdf_bytes)
        combined_text = maybe_add_ocr(pdf_bytes, extracted_text, ocr_max_pages, ocr_dpi)
        raw_text = combined_text

        safe_text_dump(
            os.path.join(raw_dir, f"{today}.source.txt"),
            f"PDF URL (auto): {pdf_url}\nSHA256: {sha256_bytes(pdf_bytes)}\n"
            f"Text chars: {len(extracted_text)} | Combined chars: {len(combined_text)}\n",
        )

    elif typ == "direct_pdf":
        pdf_url = company["pdf_url"]
        source_url = pdf_url

        pdf_resp = http_get(pdf_url)
        pdf_resp.raise_for_status()
        pdf_bytes = pdf_resp.content

        extracted_text = pdf_to_text(pdf_bytes)
        combined_text = maybe_add_ocr(pdf_bytes, extracted_text, ocr_max_pages, ocr_dpi)
        raw_text = combined_text

        safe_text_dump(
            os.path.join(raw_dir, f"{today}.source.txt"),
            f"PDF URL: {pdf_url}\nSHA256: {sha256_bytes(pdf_bytes)}\n"
            f"Text chars: {len(extracted_text)} | Combined chars: {len(combined_text)}\n",
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
                f"Text chars: {len(extracted_text)} | Combined chars: {len(combined_text)}\n",
            )
        else:
            raw_text = html_to_text(r.text)
            safe_text_dump(os.path.join(raw_dir, f"{today}.source.txt"), f"SEC URL (HTML): {pres_url}\n")

    else:
        raise RuntimeError(f"Unknown company type: {typ}")

    safe_text_dump(os.path.join(raw_dir, f"{today}.extracted_text_preview.txt"), raw_text[:20000])

    # LLM extraction
    input_cap = 160000
    extracted = llm_extract_pipeline(company["name"], raw_text[:input_cap], llm_model)
    extracted["_meta"] = {
        "run_date_utc": today,
        "source_url": source_url,
        "input_text_sha256": sha256_bytes(raw_text.encode("utf-8", errors="ignore")),
        "ocr_max_pages": ocr_max_pages,
        "ocr_dpi": ocr_dpi,
        "source_type": typ,
    }

    dated_path = os.path.join(snap_dir, f"{today}.json")
    safe_json_dump(dated_path, extracted)
    safe_json_dump(prev_path, extracted)

    if extracted.get("_llm_raw_output"):
        safe_text_dump(os.path.join(raw_dir, f"{today}.llm_raw_output.txt"), extracted["_llm_raw_output"])

    # Change classification + “why it matters”
    changes = classify_changes(prev or {}, extracted)
    analyses = llm_why_it_matters(company["name"], changes, llm_model)

    report_md = render_report(company, source_url, prev_path, extracted, changes, analyses)
    report_path = os.path.join(rep_dir, f"{today}.md")
    safe_text_dump(report_path, report_md)

    # Save machine-readable change log for later aggregation
    change_json_path = os.path.join(rep_dir, f"{today}.changes.json")
    safe_json_dump(change_json_path, {"source_url": source_url, "changes": changes, "analyses": analyses})

    return report_path, extracted, changes, source_url


def main() -> None:
    with open("config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    llm_model = cfg.get("llm_model", "gemini-2.5-flash")
    companies = cfg.get("companies", [])

    print(f"Running pipeline agent for {len(companies)} companies using model={llm_model}")

    for c in companies:
        print(f"\n--- {c['name']} ---")
        report_path, snapshot, changes, url = run_company(c, llm_model)
        print(f"Source: {url}")
        print(f"Report written: {report_path}")
        print(f"Extracted programs: {len(snapshot.get('programs', []))}")
        print(f"Changes detected: {len(changes)}")


if __name__ == "__main__":
    main()
