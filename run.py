import os
import re
import json
import time
import hashlib
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
import requests
from bs4 import BeautifulSoup

import fitz  # PyMuPDF

from google import genai


# -----------------------------
# Utilities
# -----------------------------
def utc_today_str() -> str:
    return datetime.datetime.utcnow().date().isoformat()

def ensure_dir(path: str) -> None:
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

def http_get(url: str, headers: Optional[dict] = None, timeout: int = 40) -> requests.Response:
    h = headers or {}
    resp = requests.get(url, headers=h, timeout=timeout)
    return resp

def pick_best_pdf_link(links: List[str]) -> Optional[str]:
    # Prefer JNJ-Pipeline-*.pdf if present
    pipeline = [u for u in links if re.search(r"JNJ-Pipeline-.*\.pdf$", u, re.IGNORECASE)]
    if pipeline:
        # Usually only one; if multiple, pick lexicographically last
        return sorted(set(pipeline))[-1]
    pdfs = [u for u in links if u.lower().endswith(".pdf")]
    return sorted(set(pdfs))[-1] if pdfs else None


# -----------------------------
# Source fetchers
# -----------------------------
def fetch_jnj_pipeline_pdf_url(pipeline_page_url: str) -> str:
    resp = http_get(pipeline_page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    hrefs: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            # Make absolute if needed
            if href.startswith("//"):
                href = "https:" + href
            elif href.startswith("/"):
                href = "https://www.investor.jnj.com" + href
            hrefs.append(href)

    pdf = pick_best_pdf_link(hrefs)
    if not pdf:
        raise RuntimeError("Could not find a PDF link on the J&J development pipeline page.")
    return pdf

def pdf_to_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for i in range(doc.page_count):
        parts.append(doc.load_page(i).get_text("text"))
    return "\n".join(parts)

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return norm_ws(soup.get_text(" "))

def sec_headers() -> dict:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("Missing SEC_USER_AGENT env var (set it in GitHub Secrets).")
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }

def data_sec_headers() -> dict:
    # data.sec.gov also expects declared UA; reuse same.
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("Missing SEC_USER_AGENT env var (set it in GitHub Secrets).")
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }

def sec_submissions_url(cik_10: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik_10}.json"

def accession_index_url(cik_no_leading_zeros: str, accession_no: str) -> str:
    # Example pattern described in SEC FAQs: add dashes and -index at end.
    # We'll use: /Archives/edgar/data/{cik}/{accession_no_no_dashes}/{accession_no}-index.html
    acc_nodash = accession_no.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_leading_zeros}/{acc_nodash}/{accession_no}-index.html"

def find_latest_biontech_presentation_from_sec(cik_10: str, max_filings_to_check: int = 60) -> str:
    """
    Strategy:
    - Fetch submissions JSON from data.sec.gov (no auth required per SEC docs)
    - Walk recent filings newest->oldest, focusing on 6-K
    - For each, open the filing index page and look for a document with 'presentation' in the filename
    """
    cik_int = str(int(cik_10))  # strip leading zeros for Archives path
    sub_url = sec_submissions_url(cik_10)

    # Be gentle
    r = http_get(sub_url, headers=data_sec_headers())
    r.raise_for_status()
    data = r.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    n = min(len(forms), max_filings_to_check)
    for i in range(n):
        form = forms[i]
        accession = accession_numbers[i]
        primary = primary_docs[i]
        fdate = filing_dates[i] if i < len(filing_dates) else None

        if form != "6-K":
            continue

        idx_url = accession_index_url(cik_int, accession)

        try:
            idx = http_get(idx_url, headers=sec_headers())
            if idx.status_code != 200:
                continue
            soup = BeautifulSoup(idx.text, "html.parser")
            candidates: List[str] = []
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                # Many exhibits are .htm; prefer those that look like presentations
                if re.search(r"presentation", href, re.IGNORECASE):
                    if href.startswith("http"):
                        candidates.append(href)
                    else:
                        candidates.append("https://www.sec.gov" + href)
            if candidates:
                # Pick first candidate; usually there's one
                return candidates[0]

            # Fallback: use primary document if no obvious presentation
            # Build primary doc URL
            acc_nodash = accession.replace("-", "")
            primary_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary}"
            # Use a heuristic: only accept if it looks like a presentation-ish html
            if re.search(r"(ex99|presentation|deck|slides|investor)", primary, re.IGNORECASE):
                return primary_url

        finally:
            # SEC rate limit guidance: stay well under 10 req/s
            time.sleep(0.25)

    raise RuntimeError("Could not locate a recent BioNTech presentation from SEC filings.")


# -----------------------------
# LLM extraction + change detection
# -----------------------------
def gemini_client() -> genai.Client:
    # SDK picks up GEMINI_API_KEY from env var by default
    return genai.Client()

def llm_extract_pipeline(company_name: str, source_text: str, model: str) -> dict:
    """
    Use Gemini to convert messy extracted text into a structured pipeline table.
    Output must be valid JSON (we parse it).
    """
    prompt = f"""
You are a pharmaceutical competitive intelligence analyst.

TASK:
Convert the following raw pipeline source text for {company_name} into JSON.

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
- Return valid JSON only (no markdown).
- If you are unsure about a field, set it to null.
- Phase should be one of: "Preclinical", "Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Registration", "Approved".
- Do not invent assets. Only extract what is present in the text.
- Try to capture trial names in trial_or_program if present (e.g., items in parentheses).

RAW TEXT:
{source_text}
""".strip()

    client = gemini_client()
    resp = client.models.generate_content(model=model, contents=prompt)
    text = resp.text or ""
    try:
        return json.loads(text)
    except Exception as e:
        # Save the raw model output for debugging
        return {
            "as_of_date": None,
            "programs": [],
            "_llm_parse_error": str(e),
            "_llm_raw_output": text[:8000],
        }

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

    added = [curr_map[k] for k in added_keys]
    removed = [prev_map[k] for k in removed_keys]

    return {
        "added": added,
        "removed": removed,
        "phase_changes": phase_changes,
    }

def confidence_score(diff: dict) -> float:
    # Simple heuristic v1:
    # - Any phase change = higher confidence than raw adds/removes
    score = 0.35
    if diff.get("phase_changes"):
        score += 0.35
    if diff.get("added") or diff.get("removed"):
        score += 0.20
    return max(0.0, min(0.95, score))

def render_report(company: dict, source_url: str, prev_snapshot_path: str, curr_snapshot: dict, diff: dict) -> str:
    name = company["name"]
    as_of = curr_snapshot.get("as_of_date")
    score = confidence_score(diff)

    lines = []
    lines.append(f"# Pipeline Change Report — {name}")
    lines.append("")
    lines.append(f"- Run date (UTC): {utc_today_str()}")
    lines.append(f"- Source: {source_url}")
    lines.append(f"- Extracted 'as of' date: {as_of if as_of else 'Unknown'}")
    lines.append(f"- Confidence score (0–1): {score:.2f}")
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
            if ind: bits.append(ind)
            if phase: bits.append(f"({phase})")
            if trial: bits.append(f"Trial/Program: {trial}")
            if partner: bits.append(f"Partner: {partner}")
            lines.append("- " + " — ".join(bits))
        lines.append("")

    section("New / Added items", diff.get("added", []))
    section("Removed / Discontinued items", diff.get("removed", []))

    lines.append("## Phase changes")
    pcs = diff.get("phase_changes", [])
    if not pcs:
        lines.append("_None detected._\n")
    else:
        for c in pcs:
            lines.append(f"- **{c.get('asset')}** — {c.get('indication') or ''} — {c.get('from_phase')} → {c.get('to_phase')}")
        lines.append("")

    lines.append("## Citations")
    lines.append(f"- {source_url}")
    lines.append(f"- Previous snapshot file: {prev_snapshot_path if os.path.exists(prev_snapshot_path) else 'None (first run)'}")
    lines.append("")

    return "\n".join(lines)


# -----------------------------
# Main runner
# -----------------------------
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

    if typ == "direct_pdf":
    pdf_url = company["pdf_url"]
    source_url = pdf_url
    pdf = http_get(pdf_url).content
    safe_text_dump(os.path.join(raw_dir, f"{today}.source.txt"),
                   f"PDF URL: {pdf_url}\nSHA256: {sha256_bytes(pdf)}\n")
    raw_text = pdf_to_text(pdf)

    elif typ == "sec_latest_presentation":
    cik = company["cik"]
    pres_url = find_latest_biontech_presentation_from_sec(cik)
    source_url = pres_url

        r = http_get(pres_url, headers=sec_headers())
        r.raise_for_status()

        if pres_url.lower().endswith(".pdf"):
            raw_text = pdf_to_text(r.content)
        else:
            raw_text = html_to_text(r.text)

        safe_text_dump(os.path.join(raw_dir, f"{today}.source.txt"), f"SEC URL: {pres_url}\n")
    else:
        raise RuntimeError(f"Unknown company type: {typ}")

    # LLM extraction
    extracted = llm_extract_pipeline(company["name"], raw_text[:120000], llm_model)  # cap input
    extracted["_meta"] = {
        "run_date_utc": today,
        "source_url": source_url,
        "input_text_sha256": sha256_bytes(raw_text.encode("utf-8", errors="ignore")),
    }

    # Save dated snapshot + update latest
    dated_path = os.path.join(snap_dir, f"{today}.json")
    safe_json_dump(dated_path, extracted)
    safe_json_dump(prev_path, extracted)

    # Diff + report
    d = diff_programs(prev or {}, extracted)
    report_md = render_report(company, source_url, prev_path, extracted, d)
    report_path = os.path.join(rep_dir, f"{today}.md")
    safe_text_dump(report_path, report_md)

    return report_path, extracted, d, source_url

def main():
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
        print(f"Added: {len(d.get('added', []))} | Removed: {len(d.get('removed', []))} | Phase changes: {len(d.get('phase_changes', []))}")

if __name__ == "__main__":
    main()
