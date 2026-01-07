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
