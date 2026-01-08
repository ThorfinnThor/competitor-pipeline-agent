import os
import re
import io
import csv
import json
import time
import html
import hashlib
import datetime
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse

import yaml
import requests
import xml.etree.ElementTree as ET
import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

from google import genai
from google.genai import types

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from xml.sax.saxutils import escape as xml_escape
from html.parser import HTMLParser


# -------------------------
# Utilities
# -------------------------

DATE_JSON_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")
DATE_PDF_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.pdf$")
ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$", re.ASCII)

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

def deep_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj))

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def canon(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[\W_]+", " ", s)
    return norm_ws(s)

def program_count(snapshot: Optional[dict]) -> int:
    if not snapshot:
        return 0
    return len(snapshot.get("programs", []) or [])

def http_get(url: str, headers: Optional[dict] = None, timeout: int = 60) -> requests.Response:
    return requests.get(url, headers=headers or {}, timeout=timeout)

def http_head_ok(url: str, timeout: int = 30) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def find_latest_valid_snapshot(snap_dir: str) -> Tuple[Optional[dict], Optional[str]]:
    if not os.path.isdir(snap_dir):
        return None, None
    dated = [fn for fn in os.listdir(snap_dir) if DATE_JSON_RE.match(fn)]
    dated.sort(reverse=True)
    for fn in dated:
        p = os.path.join(snap_dir, fn)
        s = safe_json_load(p)
        if program_count(s) > 0:
            return s, p
    return None, None

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


# -------------------------
# J&J Q4 CDN auto-discovery
# -------------------------

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


# -------------------------
# PDF extraction + OCR fallback
# -------------------------

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


# -------------------------
# Gemini helpers
# -------------------------

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

def _is_quota_or_rate_error(e: Exception) -> bool:
    s = str(e)
    return (
        "RESOURCE_EXHAUSTED" in s
        or "Quota exceeded" in s
        or "free_tier_requests" in s
        or "free_tier_input_token_count" in s
        or "429" in s
    )

def _is_overloaded_error(e: Exception) -> bool:
    s = str(e)
    return ("503" in s and "UNAVAILABLE" in s) or "The model is overloaded" in s

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
                if _is_quota_or_rate_error(e):
                    break
                if _is_overloaded_error(e):
                    time.sleep(2 * attempt)
                    continue
                time.sleep(1 * attempt)

    raise RuntimeError(f"All Gemini attempts failed. Last error: {last_err}")

def gemini_generate_vision(models: List[str], prompt: str, image_bytes: bytes, max_attempts_per_model: int = 2) -> str:
    client = gemini_client()
    last_err: Optional[Exception] = None

    for m in models:
        for attempt in range(1, max_attempts_per_model + 1):
            try:
                img_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                resp = client.models.generate_content(model=m, contents=[img_part, prompt])
                return (resp.text or "").strip()
            except Exception as e:
                last_err = e
                if _is_quota_or_rate_error(e):
                    break
                if _is_overloaded_error(e):
                    time.sleep(2 * attempt)
                    continue
                time.sleep(1 * attempt)

    raise RuntimeError(f"All Gemini vision attempts failed. Last error: {last_err}")


# -------------------------
# Vision pipeline extraction
# -------------------------

def render_page_png(pdf_doc: fitz.Document, page_index: int, dpi: int) -> bytes:
    page = pdf_doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")

def llm_extract_pipeline_from_page_image(company_name: str, page_no_1based: int, png_bytes: bytes, models: List[str]) -> dict:
    prompt = f"""
You are a pharma competitive intelligence analyst.

Extract all pipeline programs visible on this page image for {company_name}.
Phase is encoded visually (color and/or layout). Infer the phase from the page itself.
If you cannot determine phase with confidence, set phase to null (do NOT guess).

Return ONLY valid JSON with schema:
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
""".strip()

    out = gemini_generate_vision(models=models, prompt=prompt, image_bytes=png_bytes, max_attempts_per_model=2)
    parsed = _try_parse_json(out)
    if parsed is None:
        return {"as_of_date": None, "programs": [], "_llm_parse_error": True, "_llm_raw": out[:4000]}
    parsed.setdefault("as_of_date", None)
    parsed.setdefault("programs", [])
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
            if not merged[k].get("phase") and p.get("phase"):
                merged[k]["phase"] = p.get("phase")
            if merged[k].get("source_page") and p.get("source_page"):
                merged[k]["source_page"] = min(merged[k]["source_page"], p["source_page"])

    return {"as_of_date": as_of, "programs": list(merged.values())}

def llm_extract_pipeline_text(company_name: str, source_text: str, models: List[str]) -> dict:
    prompt = f"""
Extract a development pipeline list for {company_name} from the raw text.

Return ONLY valid JSON:
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
- Do not invent assets.
- If phase is not explicit, set phase to null.
RAW TEXT:
{source_text}
""".strip()

    out = gemini_generate_text(models=models, prompt=prompt, max_attempts_per_model=2)
    parsed = _try_parse_json(out)
    if parsed is None:
        return {"as_of_date": None, "programs": [], "_llm_parse_error": True, "_llm_raw": out[:4000]}
    parsed.setdefault("as_of_date", None)
    parsed.setdefault("programs", [])
    return parsed


# -------------------------
# Diff logic
# -------------------------

def diff_programs(prev: dict, curr: dict) -> dict:
    prev_programs = prev.get("programs", []) if prev else []
    curr_programs = curr.get("programs", []) if curr else []

    def k(p: dict) -> str:
        return canon(p.get("asset")) + "||" + canon(p.get("indication"))

    prev_map = {k(p): p for p in prev_programs if p.get("asset")}
    curr_map = {k(p): p for p in curr_programs if p.get("asset")}

    added_keys = sorted(set(curr_map.keys()) - set(prev_map.keys()))
    removed_keys = sorted(set(prev_map.keys()) - set(curr_map.keys()))
    common_keys = sorted(set(prev_map.keys()) & set(curr_map.keys()))

    phase_changes = []
    for kk in common_keys:
        p0 = prev_map[kk]
        p1 = curr_map[kk]
        if canon(p0.get("phase")) != canon(p1.get("phase")):
            phase_changes.append({
                "asset": p1.get("asset"),
                "indication": p1.get("indication"),
                "from_phase": p0.get("phase"),
                "to_phase": p1.get("phase"),
                "source_page": p1.get("source_page"),
            })

    return {
        "added": [curr_map[x] for x in added_keys],
        "removed": [prev_map[x] for x in removed_keys],
        "phase_changes": phase_changes,
    }

def phase_counts(programs: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in programs:
        ph = p.get("phase") or "Unknown"
        out[ph] = out.get(ph, 0) + 1
    return out


# -------------------------
# ClinicalTrials.gov corroboration (improved matching)
# -------------------------

CTG_BASE = "https://clinicaltrials.gov/api/v2"

def ctgov_phase_to_canon(ph: Any) -> Optional[str]:
    if not ph:
        return None
    if isinstance(ph, list) and ph:
        ranked = [ctgov_phase_to_canon(x) for x in ph]
        ranked = [x for x in ranked if x]
        if not ranked:
            return None
        rank_map = {"Preclinical": 0, "Phase 1": 1, "Phase 1/2": 2, "Phase 2": 3, "Phase 2/3": 4, "Phase 3": 5, "Registration": 6, "Approved": 7}
        ranked.sort(key=lambda x: rank_map.get(x, -1), reverse=True)
        return ranked[0]

    s = str(ph).strip().upper()
    if "PHASE1/PHASE2" in s or "PHASE 1/2" in s:
        return "Phase 1/2"
    if "PHASE2/PHASE3" in s or "PHASE 2/3" in s:
        return "Phase 2/3"
    if "PHASE 1" in s or s == "PHASE1":
        return "Phase 1"
    if "PHASE 2" in s or s == "PHASE2":
        return "Phase 2"
    if "PHASE 3" in s or s == "PHASE3":
        return "Phase 3"
    if "NA" == s or "NOT_APPLICABLE" in s:
        return None
    return None

def ctgov_get_json(params: dict) -> dict:
    r = requests.get(f"{CTG_BASE}/studies", params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def ctgov_extract_study_min(study: dict) -> dict:
    ps = (study.get("protocolSection") or {})
    ident = (ps.get("identificationModule") or {})
    status = (ps.get("statusModule") or {})
    design = (ps.get("designModule") or {})
    cond = (ps.get("conditionsModule") or {})
    spons = (ps.get("sponsorCollaboratorsModule") or {})
    arms = (ps.get("armsInterventionsModule") or {})

    nct = ident.get("nctId")
    title = ident.get("officialTitle") or ident.get("briefTitle")
    overall = status.get("overallStatus")
    last_upd = None
    for k in ["lastUpdatePostDateStruct", "lastUpdatePostDate", "studyFirstPostDateStruct"]:
        v = status.get(k)
        if isinstance(v, dict) and v.get("date"):
            last_upd = v.get("date")
            break
        if isinstance(v, str) and v:
            last_upd = v
            break

    phases = design.get("phases")
    phase_c = ctgov_phase_to_canon(phases)

    conditions = cond.get("conditions") or []
    lead = (spons.get("leadSponsor") or {}).get("name")
    collabs = [c.get("name") for c in (spons.get("collaborators") or []) if c.get("name")]

    interventions = []
    for it in (arms.get("interventions") or []):
        nm = it.get("name")
        if nm:
            interventions.append(nm)

    return {
        "nct_id": nct,
        "title": title,
        "overall_status": overall,
        "phase": phase_c,
        "conditions": conditions[:10],
        "lead_sponsor": lead,
        "collaborators": collabs[:10],
        "interventions": interventions[:10],
        "last_update": last_upd,
        "ctgov_url": f"https://clinicaltrials.gov/study/{nct}" if nct else None,
    }

def ctgov_matches_for_term(term: str, page_size: int) -> dict:
    params = {"query.term": term, "pageSize": page_size, "countTotal": "true"}
    return ctgov_get_json(params)

def sponsor_match_text(study_min: dict) -> str:
    return " ".join([
        study_min.get("lead_sponsor") or "",
        " ".join(study_min.get("collaborators") or []),
    ])

def score_study_for_terms(study_min: dict, terms: List[str], sponsor_keywords: List[str]) -> float:
    score = 0.0
    title = (study_min.get("title") or "")
    interventions = " ".join(study_min.get("interventions") or [])
    spons = sponsor_match_text(study_min)

    # sponsor boosts
    spons_l = spons.lower()
    for kw in sponsor_keywords or []:
        if kw.lower() in spons_l:
            score += 2.0
            break

    # term hits
    hay = f"{title} {interventions}".lower()
    for t in terms:
        tl = t.lower()
        if not tl or len(tl) < 3:
            continue
        if tl in hay:
            score += 2.0
        elif tl in title.lower():
            score += 1.0

    # recency mild boost (if last_update exists)
    lu = study_min.get("last_update") or ""
    if isinstance(lu, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", lu):
        try:
            d = datetime.date.fromisoformat(lu)
            delta = (datetime.datetime.utcnow().date() - d).days
            if delta <= 60:
                score += 0.5
            if delta <= 14:
                score += 0.5
        except Exception:
            pass

    return score

def extract_query_terms_from_asset(asset: str) -> List[str]:
    """
    Build multiple query terms:
    - full asset string
    - pieces inside parentheses
    - split on '/' and ',' and ' - '
    """
    asset = norm_ws(asset)
    out = []
    if asset:
        out.append(asset)

    # parentheses aliases
    aliases = re.findall(r"\(([^)]+)\)", asset)
    for a in aliases:
        a = norm_ws(a)
        if a:
            out.append(a)

    # split on separators
    pieces = re.split(r"[\/,;]| - ", asset)
    for p in pieces:
        p = norm_ws(re.sub(r"\([^)]*\)", "", p))
        if p and p.lower() not in ("jnj", "janssen"):
            out.append(p)

    # keep unique by canon
    seen = set()
    uniq = []
    for t in out:
        ct = canon(t)
        if ct and ct not in seen:
            seen.add(ct)
            uniq.append(t)
    return uniq[:5]  # cap to control API usage

def ctgov_corroborate(snapshot: dict, diff: dict, cfg_company: dict, run_date: str, sources_dir: str) -> dict:
    if not cfg_company.get("ctgov_enabled", True):
        return {"enabled": False}

    page_size = int(cfg_company.get("ctgov_page_size", 20))
    max_assets = int(cfg_company.get("ctgov_max_assets_per_run", 40))
    sponsor_keywords = cfg_company.get("ctgov_sponsor_keywords") or []

    # prioritize changed assets
    priority_assets: List[str] = []
    for it in (diff.get("added") or []):
        if it.get("asset"):
            priority_assets.append(it["asset"])
    for it in (diff.get("phase_changes") or []):
        if it.get("asset"):
            priority_assets.append(it["asset"])

    # if no changes, sample top N
    if not priority_assets:
        for p in (snapshot.get("programs") or [])[:max_assets]:
            if p.get("asset"):
                priority_assets.append(p["asset"])

    # dedupe
    seen = set()
    assets = []
    for a in priority_assets:
        ca = canon(a)
        if ca and ca not in seen:
            seen.add(ca)
            assets.append(a)
        if len(assets) >= max_assets:
            break

    run_dir = os.path.join(sources_dir, "ctgov", run_date)
    ensure_dir(run_dir)

    results_by_asset: Dict[str, dict] = {}
    total_queries = 0

    for asset in assets:
        terms = extract_query_terms_from_asset(asset)
        combined: Dict[str, dict] = {}  # by nct_id
        queries = []

        for term in terms:
            total_queries += 1
            queries.append(term)
            try:
                raw = ctgov_matches_for_term(term, page_size=page_size)
                studies = raw.get("studies") or []
                studies_min = [ctgov_extract_study_min(s) for s in studies]
                for smin in studies_min:
                    nct = smin.get("nct_id") or f"NOID_{sha256_bytes((smin.get('title','') or '').encode('utf-8'))[:12]}"
                    if nct not in combined:
                        combined[nct] = smin
            except Exception:
                continue
            time.sleep(0.15)

        # score & select
        scored = []
        for smin in combined.values():
            sc = score_study_for_terms(smin, terms, sponsor_keywords)
            scored.append((sc, smin))
        scored.sort(key=lambda x: x[0], reverse=True)

        chosen = [s for sc, s in scored[:10] if sc > 0.5] or [s for sc, s in scored[:5]]

        store_obj = {
            "asset": asset,
            "query_terms": queries,
            "returned_unique": len(combined),
            "kept": len(chosen),
            "studies": chosen,
        }
        safe_fn = re.sub(r"[^A-Za-z0-9._-]+", "_", asset)[:80]
        safe_json_dump(os.path.join(run_dir, f"{safe_fn}.json"), store_obj)
        results_by_asset[asset] = store_obj

    # attach to programs
    prog_list = snapshot.get("programs") or []
    by_asset_canon: Dict[str, List[dict]] = {}
    for asset, res in results_by_asset.items():
        by_asset_canon.setdefault(canon(asset), []).extend(res.get("studies") or [])

    corroborated = 0
    mismatches = 0
    mismatch_list = []

    for p in prog_list:
        a = p.get("asset") or ""
        matches = by_asset_canon.get(canon(a), [])
        p["_ctgov_matches"] = matches
        p["_ctgov_query_terms"] = extract_query_terms_from_asset(a)

        if matches:
            corroborated += 1
            reg_phase = ctgov_phase_to_canon([m.get("phase") for m in matches if m.get("phase")])
            pipe_phase = p.get("phase")
            p["_ctgov_best_phase"] = reg_phase
            if reg_phase and pipe_phase and canon(reg_phase) != canon(pipe_phase):
                mismatches += 1
                p["_ctgov_phase_flag"] = "mismatch"
                mismatch_list.append({
                    "asset": p.get("asset"),
                    "indication": p.get("indication"),
                    "pipeline_phase": pipe_phase,
                    "ctgov_phase": reg_phase,
                })
            else:
                p["_ctgov_phase_flag"] = "ok"
        else:
            p["_ctgov_best_phase"] = None
            p["_ctgov_phase_flag"] = "none"

    summary = {
        "enabled": True,
        "queried_assets": assets,
        "queries_executed": total_queries,
        "programs_total": len(prog_list),
        "programs_with_matches": corroborated,
        "phase_mismatches": mismatches,
        "phase_mismatch_examples": mismatch_list[:20],
        "sources_path": run_dir,
        "api_base": CTG_BASE,
    }
    snapshot.setdefault("_meta", {})
    snapshot["_meta"]["ctgov"] = summary
    return summary


# -------------------------
# SEC EDGAR corroboration (improved lookback + dedupe)
# -------------------------

def sec_headers() -> dict:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("Missing SEC_USER_AGENT env var (should include contact info).")
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json,text/html,application/xhtml+xml",
    }

def sec_get_json(url: str) -> dict:
    r = requests.get(url, headers=sec_headers(), timeout=60)
    r.raise_for_status()
    return r.json()

def sec_get_bytes(url: str) -> bytes:
    r = requests.get(url, headers=sec_headers(), timeout=60)
    r.raise_for_status()
    return r.content

def sec_cik_norm(cik10: str) -> Tuple[str, str]:
    c = re.sub(r"\D", "", cik10 or "")
    c = c.zfill(10)
    return c, str(int(c))

def strip_html_to_text(b: bytes) -> str:
    s = b.decode("utf-8", errors="ignore")
    s = re.sub(r"(?is)<script.*?>.*?</script>", " ", s)
    s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
    s = re.sub(r"(?is)<br\s*/?>", "\n", s)
    s = re.sub(r"(?is)</p\s*>", "\n\n", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def sec_recent_filings(cik10: str, since_date: str, forms: List[str], max_filings: int) -> List[dict]:
    cik10, cik_int = sec_cik_norm(cik10)
    sub = sec_get_json(f"https://data.sec.gov/submissions/CIK{cik10}.json")
    recent = ((sub.get("filings") or {}).get("recent") or {})

    forms_arr = recent.get("form") or []
    acc_arr = recent.get("accessionNumber") or []
    date_arr = recent.get("filingDate") or []
    prim_arr = recent.get("primaryDocument") or []
    desc_arr = recent.get("primaryDocDescription") or []

    out = []
    for i in range(min(len(forms_arr), len(acc_arr), len(date_arr), len(prim_arr))):
        form = forms_arr[i]
        if forms and form not in forms:
            continue
        fd = date_arr[i]
        if since_date and fd < since_date:
            continue
        out.append({
            "form": form,
            "filing_date": fd,
            "accession": acc_arr[i],
            "primary_document": prim_arr[i],
            "primary_desc": desc_arr[i] if i < len(desc_arr) else None,
            "cik10": cik10,
            "cik_int": cik_int,
        })

    out.sort(key=lambda x: x["filing_date"], reverse=True)
    return out[:max_filings]

def load_processed_accessions(sec_sources_dir: str) -> set:
    processed = set()
    if not os.path.isdir(sec_sources_dir):
        return processed
    # Expected: sources/jnj/sec/<run_date>/<accession>.events.json
    for root, _, files in os.walk(sec_sources_dir):
        for fn in files:
            if fn.endswith(".events.json"):
                acc = fn.replace(".events.json", "")
                if ACCESSION_RE.match(acc):
                    processed.add(acc)
    return processed

def llm_extract_edgar_events(company_name: str, filing_meta: dict, filing_text: str, models: List[str]) -> dict:
    prompt = f"""
You are a senior pharma competitive intelligence analyst.

From the SEC filing text below, extract ONLY pipeline-relevant signals for {company_name}.
Focus on: discontinuations, prioritization, partnerships, clinical phase progress, regulatory filings, trial outcomes.

Return ONLY valid JSON:
{{
  "filing": {{
    "form": string,
    "filing_date": string,
    "accession": string,
    "url": string
  }},
  "events": [
    {{
      "event_type": "partnership|discontinuation|phase_progression|regulatory|other",
      "asset": string | null,
      "indication": string | null,
      "summary": string,
      "confidence": number
    }}
  ]
}}

Rules:
- Do not invent assets.
- Confidence in [0,1].
- If there are no clear pipeline-relevant signals, return an empty events array.
""".strip()

    try:
        out = gemini_generate_text(
            models=models,
            prompt=prompt
            + "\n\nFILING_META:\n" + json.dumps(filing_meta, ensure_ascii=False)
            + "\n\nFILING_TEXT:\n" + filing_text[:120000],
            max_attempts_per_model=2
        )
        parsed = _try_parse_json(out)
        if parsed:
            parsed.setdefault("filing", filing_meta)
            parsed.setdefault("events", [])
            return parsed
        return {"filing": filing_meta, "events": [], "_llm_parse_error": True, "_llm_raw": (out or "")[:2000]}
    except Exception as e:
        return {"filing": filing_meta, "events": [], "_llm_error": str(e)}

def edgar_corroborate(snapshot: dict, cfg_company: dict, models: List[str], run_date: str, sources_dir: str, since_date: str) -> dict:
    if not cfg_company.get("sec_enabled", True):
        return {"summary": {"enabled": False}, "filing_events": []}

    cik10 = str(cfg_company.get("sec_cik_10digit", "")).strip()
    forms = cfg_company.get("sec_forms") or ["10-K", "10-Q", "8-K"]
    max_filings = int(cfg_company.get("sec_max_filings_per_run", 6))
    store_max_bytes = int(cfg_company.get("sec_store_max_bytes", 2000000))

    sec_root = os.path.join(sources_dir, "sec")
    processed = load_processed_accessions(sec_root)

    filings = sec_recent_filings(cik10=cik10, since_date=since_date, forms=forms, max_filings=max_filings * 2)
    # filter already processed
    filings = [f for f in filings if f.get("accession") not in processed][:max_filings]

    run_dir = os.path.join(sec_root, run_date)
    ensure_dir(run_dir)

    extracted_events: List[dict] = []
    considered = 0

    for fmeta in filings:
        considered += 1
        acc = fmeta["accession"]
        cik_int = fmeta["cik_int"]
        acc_nodash = acc.replace("-", "")
        primary = fmeta["primary_document"]
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary}"
        fmeta["url"] = url

        try:
            b = sec_get_bytes(url)
            excerpt = b[:store_max_bytes]

            safe_text_dump(os.path.join(run_dir, f"{acc}.url.txt"), url)
            safe_text_dump(os.path.join(run_dir, f"{acc}.sha256.txt"), sha256_bytes(b))
            with open(os.path.join(run_dir, f"{acc}.excerpt.bin"), "wb") as fh:
                fh.write(excerpt)

            filing_text = strip_html_to_text(b)
            safe_text_dump(os.path.join(run_dir, f"{acc}.text.txt"), filing_text[:250000])

            ev = llm_extract_edgar_events(cfg_company.get("name", "Company"), fmeta, filing_text, models=models)
            safe_json_dump(os.path.join(run_dir, f"{acc}.events.json"), ev)
            extracted_events.append(ev)

        except Exception as e:
            safe_json_dump(os.path.join(run_dir, f"{acc}.error.json"), {"filing": fmeta, "error": str(e)})

        time.sleep(0.25)

    summary = {
        "enabled": True,
        "since_date": since_date,
        "forms": forms,
        "filings_considered": considered,
        "filings_with_events": sum(1 for e in extracted_events if (e.get("events") or [])),
        "sources_path": run_dir,
        "deduped_accessions_count": len(processed),
    }

    snapshot.setdefault("_meta", {})
    snapshot["_meta"]["sec_edgar"] = summary
    return {"summary": summary, "filing_events": extracted_events}


# -------------------------
# Press release ingestion (RSS-first, 403-safe)
# -------------------------

import xml.etree.ElementTree as ET

DEFAULT_WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

def press_fetch_index(feed_url: str) -> str:
    """
    Fetch a press feed index safely.
    - Uses browser-like headers.
    - If the IR site blocks (403), we will let the caller handle fallback/disable.
    """
    r = requests.get(feed_url, headers=DEFAULT_WEB_HEADERS, timeout=60)
    r.raise_for_status()
    return r.text

def _looks_like_rss(text: str) -> bool:
    t = (text or "").lstrip().lower()
    return t.startswith("<?xml") or "<rss" in t[:500] or "<feed" in t[:500]

def _normalize_pubdate_to_yyyy_mm_dd(pub: str) -> Optional[str]:
    pub = (pub or "").strip()
    if not pub:
        return None

    # Common RSS pubDate: "Mon, 26 Dec 2025 14:00:00 -0500"
    fmts = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%Z",
        "%Y-%m-%d",
    ]
    for f in fmts:
        try:
            dt = datetime.datetime.strptime(pub, f)
            return dt.date().isoformat()
        except Exception:
            continue
    return None

def press_parse_rss_items(feed_url: str, xml_text: str, max_items: int = 10) -> List[dict]:
    """
    Parse RSS or Atom feed. Return list of dicts:
      { "url": ..., "title": ..., "date": ... }
    """
    items: List[dict] = []
    try:
        root = ET.fromstring(xml_text.encode("utf-8", errors="ignore"))
    except Exception:
        return items

    # RSS: <rss><channel><item>...
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub = (it.findtext("pubDate") or "").strip()
        date_norm = _normalize_pubdate_to_yyyy_mm_dd(pub)
        if link:
            items.append({"url": urljoin(feed_url, link), "title": title or None, "date": date_norm})
        if len(items) >= max_items:
            return items

    # Atom: <feed><entry>...
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for ent in root.findall(".//a:entry", ns):
        title = (ent.findtext("a:title", default="", namespaces=ns) or "").strip()
        pub = (ent.findtext("a:updated", default="", namespaces=ns) or "").strip()
        date_norm = _normalize_pubdate_to_yyyy_mm_dd(pub[:10]) or _normalize_pubdate_to_yyyy_mm_dd(pub)

        link = None
        for l in ent.findall("a:link", ns):
            href = l.attrib.get("href")
            rel = l.attrib.get("rel", "alternate")
            if href and rel == "alternate":
                link = href
                break
            if href and not link:
                link = href

        if link:
            items.append({"url": urljoin(feed_url, link), "title": title or None, "date": date_norm})
        if len(items) >= max_items:
            return items

    return items

def press_url_to_safe_name(url: str) -> str:
    p = urlparse(url)
    s = (p.path or "").strip("/").replace("/", "_")
    if not s:
        s = "press_release"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120]
    return s

def press_html_to_text(html_bytes: bytes) -> str:
    return strip_html_to_text(html_bytes)

def press_keyword_events(text: str) -> List[dict]:
    """Deterministic fallback if LLM is unavailable/quota-limited."""
    t = (text or "").lower()
    events = []
    keywords = [
        ("partnership", ["collaboration", "partnership", "agreement", "license", "acquire", "acquisition"]),
        ("discontinuation", ["discontinue", "terminated", "termination", "stop development", "halt"]),
        ("regulatory", ["fda", "ema", "approval", "submission", "nda", "bla", "marketing authorization"]),
        ("phase_progression", ["phase 3", "phase 2", "phase 1", "registrational", "pivotal", "topline"]),
    ]
    for etype, kws in keywords:
        for kw in kws:
            if kw in t:
                events.append({
                    "event_type": etype,
                    "asset": None,
                    "indication": None,
                    "summary": f"Keyword signal detected in press release text: '{kw}'. Review stored press release for details.",
                    "confidence": 0.25,
                })
                break
    return events[:6]

def llm_extract_press_events(company_name: str, url: str, title: str, date_guess: str, text: str, models: List[str]) -> dict:
    prompt = f"""
You are a senior pharma competitive intelligence analyst.

From the press release text for {company_name}, extract ONLY pipeline-relevant signals.
Focus on: partnerships, discontinuations, phase progression, regulatory milestones, trial results.

Return ONLY valid JSON:
{{
  "press_release": {{
    "url": string,
    "title": string,
    "date": string | null
  }},
  "events": [
    {{
      "event_type": "partnership|discontinuation|phase_progression|regulatory|other",
      "asset": string | null,
      "indication": string | null,
      "summary": string,
      "confidence": number
    }}
  ]
}}

Rules:
- Do not invent assets.
- Confidence in [0,1].
- If nothing relevant, return empty events array.
""".strip()

    meta = {"url": url, "title": title, "date": date_guess or None}

    try:
        out = gemini_generate_text(
            models=models,
            prompt=prompt + "\n\nMETA:\n" + json.dumps(meta, ensure_ascii=False) + "\n\nTEXT:\n" + (text or "")[:120000],
            max_attempts_per_model=2
        )
        parsed = _try_parse_json(out)
        if parsed:
            parsed.setdefault("press_release", meta)
            parsed.setdefault("events", [])
            return parsed
        return {"press_release": meta, "events": [], "_llm_parse_error": True, "_llm_raw": (out or "")[:2000]}
    except Exception as e:
        # Never crash the run because press is auxiliary
        return {"press_release": meta, "events": press_keyword_events(text or ""), "_llm_error": str(e)}

def load_processed_press_urls(press_root: str) -> set:
    processed = set()
    if not os.path.isdir(press_root):
        return processed
    for root, _, files in os.walk(press_root):
        for fn in files:
            if fn.endswith(".meta.json"):
                p = os.path.join(root, fn)
                meta = safe_json_load(p) or {}
                u = meta.get("url")
                if u:
                    processed.add(u)
    return processed

def press_corroborate(snapshot: dict, cfg_company: dict, models: List[str], run_date: str, sources_dir: str) -> dict:
    """
    RSS-first press ingestion.
    - Will NOT raise if blocked (403); returns a summary with error instead.
    """
    if not cfg_company.get("press_enabled", False):
        return {"summary": {"enabled": False}, "press_events": []}

    feed_url = cfg_company.get("press_feed_url") or "https://www.jnj.com/rss-feed/all"
    max_items = int(cfg_company.get("press_max_items_per_run", 10))

    press_root = os.path.join(sources_dir, "press")
    ensure_dir(press_root)

    processed = load_processed_press_urls(press_root)

    run_dir = os.path.join(press_root, run_date)
    ensure_dir(run_dir)

    try:
        idx_text = press_fetch_index(feed_url)
        safe_text_dump(os.path.join(run_dir, "feed.url.txt"), feed_url)
        safe_text_dump(os.path.join(run_dir, "feed.body.txt"), idx_text[:250000])
    except Exception as e:
        # Do not crash the pipeline run
        summary = {
            "enabled": True,
            "feed_url": feed_url,
            "links_scanned": 0,
            "new_releases_processed": 0,
            "releases_with_events": 0,
            "sources_path": run_dir,
            "error": f"Failed to fetch press feed: {e}",
        }
        snapshot.setdefault("_meta", {})
        snapshot["_meta"]["press"] = summary
        return {"summary": summary, "press_events": []}

    # Parse RSS/Atom; if not RSS, bail safely (you can extend later)
    items = []
    if _looks_like_rss(idx_text):
        items = press_parse_rss_items(feed_url, idx_text, max_items=max_items)
    else:
        summary = {
            "enabled": True,
            "feed_url": feed_url,
            "links_scanned": 0,
            "new_releases_processed": 0,
            "releases_with_events": 0,
            "sources_path": run_dir,
            "error": "Press feed did not look like RSS/Atom. Use an RSS feed URL (recommended: https://www.jnj.com/rss-feed/all).",
        }
        snapshot.setdefault("_meta", {})
        snapshot["_meta"]["press"] = summary
        return {"summary": summary, "press_events": []}

    considered = 0
    new_items = 0
    extracted = []

    for it in items:
        url = it.get("url")
        if not url:
            continue
        considered += 1
        if url in processed:
            continue
        new_items += 1

        try:
            r = requests.get(url, headers=DEFAULT_WEB_HEADERS, timeout=60)
            r.raise_for_status()
            b = r.content
            text = press_html_to_text(b)

            safe_name = press_url_to_safe_name(url)

            # store provenance
            safe_text_dump(os.path.join(run_dir, f"{safe_name}.url.txt"), url)
            safe_text_dump(os.path.join(run_dir, f"{safe_name}.sha256.txt"), sha256_bytes(b))
            safe_text_dump(os.path.join(run_dir, f"{safe_name}.text.txt"), text[:250000])

            meta = {
                "url": url,
                "title": it.get("title") or safe_name,
                "date": it.get("date"),
            }
            safe_json_dump(os.path.join(run_dir, f"{safe_name}.meta.json"), meta)

            ev = llm_extract_press_events(cfg_company.get("name", "Company"), url, meta["title"], meta.get("date") or "", text, models=models)
            safe_json_dump(os.path.join(run_dir, f"{safe_name}.events.json"), ev)
            extracted.append(ev)

        except Exception as e:
            safe_json_dump(os.path.join(run_dir, f"{press_url_to_safe_name(url)}.error.json"), {"url": url, "error": str(e)})

        time.sleep(0.2)

    summary = {
        "enabled": True,
        "feed_url": feed_url,
        "links_scanned": considered,
        "new_releases_processed": new_items,
        "releases_with_events": sum(1 for e in extracted if (e.get("events") or [])),
        "sources_path": run_dir,
    }

    snapshot.setdefault("_meta", {})
    snapshot["_meta"]["press"] = summary
    return {"summary": summary, "press_events": extracted}



# -------------------------
# Narrative synthesis (uses diff + CT.gov + EDGAR + press)
# -------------------------

def llm_final_brief(company_name: str, diff: dict, ctgov_summary: dict, edgar_pack: dict, press_pack: dict, models: List[str]) -> dict:
    no_changes = not (diff.get("added") or diff.get("removed") or diff.get("phase_changes"))
    ed_events = (edgar_pack or {}).get("filing_events") or []
    pr_events = (press_pack or {}).get("press_events") or []

    has_edgar_events = any((b.get("events") or []) for b in ed_events)
    has_press_events = any((b.get("events") or []) for b in pr_events)

    if no_changes and not has_edgar_events and not has_press_events:
        return {
            "headline": "No pipeline update detected",
            "executive_summary": (
                "The pipeline deck appears unchanged versus the last valid snapshot, and no new pipeline-relevant signals "
                "were extracted from SEC filings or press releases in this run. Review corroboration sections and watchlist."
            ),
            "top_changes": [],
            "watchlist": [
                "Monitor ClinicalTrials.gov for phase/status updates on priority assets (registry updates can precede deck refresh).",
                "Monitor upcoming earnings materials for strategic reprioritization and portfolio actions.",
                "Monitor new SEC filings and press releases for deal activity or development discontinuations.",
            ],
        }

    payload = {
        "company": company_name,
        "diff": diff,
        "ctgov": ctgov_summary,
        "sec_edgar": (edgar_pack or {}).get("summary") or {},
        "edgar_events": ed_events,
        "press": (press_pack or {}).get("summary") or {},
        "press_events": pr_events,
    }

    prompt = f"""
You are a senior pharma competitive intelligence analyst.

Write a decision-oriented monthly competitor pipeline update for {company_name}.

Use:
- The pipeline diff as the anchor (new assets, removals, phase changes),
- ClinicalTrials.gov corroboration (support/contradiction, freshness),
- SEC EDGAR signals (disclosures),
- IR press release signals (deals, discontinuations, milestones).

Return ONLY valid JSON:
{{
  "headline": string,
  "executive_summary": string,
  "top_changes": [
    {{
      "change_type": "new_asset|removed_asset|phase_change|deal|discontinuation|regulatory|other",
      "asset": string | null,
      "indication": string | null,
      "why_it_matters": string,
      "confidence": number,
      "evidence": [string]
    }}
  ],
  "watchlist": [string]
}}

Rules:
- Confidence in [0,1], higher if corroborated by CT.gov and/or EDGAR/press.
- If no pipeline changes, focus on EDGAR/press signals + watchlist.
INPUT_JSON:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    try:
        out = gemini_generate_text(models=models, prompt=prompt, max_attempts_per_model=2)
        parsed = _try_parse_json(out)
        if parsed:
            parsed.setdefault("headline", "Pipeline monitoring update")
            parsed.setdefault("executive_summary", "")
            parsed.setdefault("top_changes", [])
            parsed.setdefault("watchlist", [])
            return parsed
    except Exception as e:
        return {
            "headline": "Pipeline monitoring update (LLM unavailable)",
            "executive_summary": (
                "The run completed, but narrative synthesis failed due to model quota/rate limits. "
                "Use the change log, inventory CSV, and evidence pack for decisions."
            ),
            "top_changes": [],
            "watchlist": [
                "Re-run when quota is available to generate a ranked narrative summary.",
                "Use evidence JSON + stored sources to validate any key asset changes.",
            ],
            "_error": str(e),
        }

    return {
        "headline": "Pipeline monitoring update (LLM output parse failed)",
        "executive_summary": "The run completed but narrative output could not be parsed as JSON. Refer to change log and evidence.",
        "top_changes": [],
        "watchlist": [],
    }


# -------------------------
# Reporting (MD + PDF)
# -------------------------

def export_programs_csv(path: str, programs: List[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asset", "indication", "phase", "trial_or_program", "partner", "notes", "source_page", "ctgov_best_phase", "ctgov_flag"])
        for p in programs:
            w.writerow([
                p.get("asset") or "",
                p.get("indication") or "",
                p.get("phase") or "",
                p.get("trial_or_program") or "",
                p.get("partner") or "",
                p.get("notes") or "",
                p.get("source_page") or "",
                p.get("_ctgov_best_phase") or "",
                p.get("_ctgov_phase_flag") or "",
            ])

def render_markdown(company: dict, run_date: str, snapshot: dict, diff: dict, final_brief: dict,
                    source_url: str, source_pdf_path: str, source_sha256: str,
                    reused_snapshot: bool, recovered_baseline: bool,
                    programs_csv_path: str, evidence_json_path: str,
                    ctgov_summary: dict, edgar_pack: dict, press_pack: dict) -> str:
    lines: List[str] = []
    name = company.get("name", "Company")

    lines.append(f"# Competitive Pipeline Report — {name}\n")

    lines.append("## Run coverage & provenance")
    lines.append(f"- Run date (UTC): {run_date}")
    lines.append(f"- Pipeline source URL: {source_url}")
    lines.append(f"- Stored pipeline PDF: `{source_pdf_path}`")
    lines.append(f"- Pipeline SHA256: `{source_sha256}`")
    lines.append(f"- Programs extracted: {len(snapshot.get('programs') or [])}")
    if recovered_baseline:
        lines.append("- Note: latest.json baseline was invalid; recovered most recent valid dated snapshot.")
    if reused_snapshot:
        lines.append("- Note: Source PDF unchanged; reused prior snapshot to avoid extraction variability.")
    lines.append(f"- Inventory export: `{programs_csv_path}`")
    lines.append(f"- Evidence pack: `{evidence_json_path}`\n")

    lines.append("## Executive summary")
    lines.append(f"**{final_brief.get('headline','Pipeline monitoring update')}**\n")
    if final_brief.get("executive_summary"):
        lines.append(final_brief["executive_summary"] + "\n")

    # Top changes
    lines.append("## Top changes (ranked)")
    tc = final_brief.get("top_changes") or []
    if not tc:
        lines.append("_No ranked changes generated._\n")
    else:
        for c in tc:
            lines.append(f"- **{c.get('change_type','')}** — {c.get('asset') or ''} — {c.get('indication') or ''} (conf: {float(c.get('confidence',0)):.2f})")
            lines.append(f"  - {c.get('why_it_matters','')}")
            ev = c.get("evidence") or []
            if ev:
                lines.append("  - Evidence:")
                for e in ev[:8]:
                    lines.append(f"    - {e}")
        lines.append("")

    # Change log
    lines.append("## Change log (previous vs current)")
    def sec(title: str, items: List[dict]):
        lines.append(f"### {title}")
        if not items:
            lines.append("_None detected._\n")
            return
        for it in items[:200]:
            a = it.get("asset","")
            ind = it.get("indication","")
            ph = it.get("phase") or ""
            pg = it.get("source_page") or ""
            lines.append(f"- **{a}** — {ind} — {ph} (p.{pg})")
        lines.append("")

    sec("New / Added items", diff.get("added") or [])
    sec("Removed / Discontinued items", diff.get("removed") or [])

    lines.append("### Phase changes")
    pcs = diff.get("phase_changes") or []
    if not pcs:
        lines.append("_None detected._\n")
    else:
        for it in pcs[:200]:
            lines.append(f"- **{it.get('asset')}** — {it.get('indication') or ''} — {it.get('from_phase')} → {it.get('to_phase')} (p.{it.get('source_page')})")
        lines.append("")

    # CT.gov
    lines.append("## ClinicalTrials.gov corroboration")
    if not ctgov_summary.get("enabled"):
        lines.append("_Disabled._\n")
    else:
        lines.append(f"- API base: {ctgov_summary.get('api_base')}")
        lines.append(f"- Programs with matches: {ctgov_summary.get('programs_with_matches')}/{ctgov_summary.get('programs_total')}")
        lines.append(f"- Phase mismatches (pipeline vs registry): {ctgov_summary.get('phase_mismatches')}")
        if ctgov_summary.get("phase_mismatch_examples"):
            lines.append("- Example mismatches:")
            for m in ctgov_summary["phase_mismatch_examples"][:10]:
                lines.append(f"  - {m.get('asset')} — pipeline: {m.get('pipeline_phase')} vs CT.gov: {m.get('ctgov_phase')}")
        lines.append(f"- Stored CT.gov evidence: `{ctgov_summary.get('sources_path')}`\n")

    # EDGAR
    lines.append("## SEC EDGAR signals")
    ed_sum = (edgar_pack or {}).get("summary") or {}
    if not ed_sum.get("enabled"):
        lines.append("_Disabled._\n")
    else:
        lines.append(f"- Filings since: {ed_sum.get('since_date')}")
        lines.append(f"- Filings considered (new): {ed_sum.get('filings_considered')}")
        lines.append(f"- Filings with extracted events: {ed_sum.get('filings_with_events')}")
        lines.append(f"- Stored EDGAR evidence: `{ed_sum.get('sources_path')}`\n")
        fe = (edgar_pack or {}).get("filing_events") or []
        for block in fe:
            filing = block.get("filing") or {}
            events = block.get("events") or []
            if not events:
                continue
            lines.append(f"### {filing.get('form')} — {filing.get('filing_date')} — {filing.get('accession')}")
            lines.append(f"- URL: {filing.get('url')}")
            for ev in events[:10]:
                lines.append(f"- **{ev.get('event_type')}** (conf {float(ev.get('confidence',0)):.2f}): {ev.get('summary','')}")
            lines.append("")

    # Press
    lines.append("## IR press release signals")
    pr_sum = (press_pack or {}).get("summary") or {}
    if not pr_sum.get("enabled"):
        lines.append("_Disabled._\n")
    else:
        lines.append(f"- Feed URL: {pr_sum.get('feed_url')}")
        lines.append(f"- Links scanned: {pr_sum.get('links_scanned')}")
        lines.append(f"- New releases processed: {pr_sum.get('new_releases_processed')}")
        lines.append(f"- Releases with extracted events: {pr_sum.get('releases_with_events')}")
        lines.append(f"- Stored press evidence: `{pr_sum.get('sources_path')}`\n")
        pe = (press_pack or {}).get("press_events") or []
        for block in pe:
            meta = block.get("press_release") or {}
            events = block.get("events") or []
            if not events:
                continue
            lines.append(f"### {meta.get('date') or ''} — {meta.get('title') or ''}")
            lines.append(f"- URL: {meta.get('url')}")
            for ev in events[:8]:
                lines.append(f"- **{ev.get('event_type')}** (conf {float(ev.get('confidence',0)):.2f}): {ev.get('summary','')}")
            lines.append("")

    # Inventory
    lines.append("## Pipeline inventory (full list)")
    lines.append("| Asset | Indication | Phase | Page | CT.gov phase | Flag |")
    lines.append("|---|---|---|---:|---|---|")
    for p in sorted(snapshot.get("programs") or [], key=lambda x: (x.get("phase") or "ZZZ", x.get("asset") or "")):
        lines.append(f"| {p.get('asset','')} | {p.get('indication','') or ''} | {p.get('phase') or ''} | {p.get('source_page') or ''} | {p.get('_ctgov_best_phase') or ''} | {p.get('_ctgov_phase_flag') or ''} |")
    lines.append("")

    lines.append("## Citations / provenance")
    lines.append(f"- Pipeline source: {source_url}")
    lines.append(f"- Stored pipeline PDF: `{source_pdf_path}`")
    lines.append(f"- SHA256: `{source_sha256}`")
    if ctgov_summary.get("enabled"):
        lines.append(f"- CT.gov evidence folder: `{ctgov_summary.get('sources_path')}`")
    if (edgar_pack or {}).get("summary", {}).get("enabled"):
        lines.append(f"- EDGAR evidence folder: `{(edgar_pack or {}).get('summary', {}).get('sources_path')}`")
    if (press_pack or {}).get("summary", {}).get("enabled"):
        lines.append(f"- Press evidence folder: `{(press_pack or {}).get('summary', {}).get('sources_path')}`")

    return "\n".join(lines)

def build_pdf(pdf_path: str, company: dict, run_date: str, snapshot: dict, diff: dict, final_brief: dict,
              source_url: str, source_pdf_path: str, source_sha256: str,
              reused_snapshot: bool, recovered_baseline: bool,
              programs_csv_path: str, evidence_json_path: str,
              ctgov_summary: dict, edgar_pack: dict, press_pack: dict) -> None:

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], spaceAfter=10))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], spaceAfter=8))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], leading=11, fontSize=9, spaceAfter=4))

    def P(txt: str, style="Bodyx") -> Paragraph:
        return Paragraph(xml_escape(txt or ""), styles[style])

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Competitive Pipeline Report",
        author="Competitor Pipeline Agent",
    )

    name = company.get("name", "Company")
    programs = snapshot.get("programs") or []
    counts = phase_counts(programs)

    story: List[Any] = []
    story.append(P(f"Competitive Pipeline Report — {name}", "H1x"))
    story.append(P(f"Run date (UTC): {run_date}", "Bodyx"))

    prov_rows = [
        ["Pipeline source URL", source_url],
        ["Stored pipeline PDF", source_pdf_path],
        ["Pipeline SHA256", source_sha256],
        ["Programs extracted", str(len(programs))],
        ["Recovered baseline", "Yes" if recovered_baseline else "No"],
        ["Doc unchanged / reused", "Yes" if reused_snapshot else "No"],
        ["Inventory CSV", programs_csv_path],
        ["Evidence pack", evidence_json_path],
    ]
    tbl = Table(prov_rows, colWidths=[1.8 * inch, 4.7 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(Spacer(1, 10))
    story.append(tbl)
    story.append(Spacer(1, 12))

    story.append(P(final_brief.get("headline", "Pipeline monitoring update"), "H2x"))
    story.append(P(final_brief.get("executive_summary", ""), "Bodyx"))

    # Top changes table
    tc = final_brief.get("top_changes") or []
    story.append(P("Top changes (ranked)", "H2x"))
    if not tc:
        story.append(P("No ranked changes generated.", "Bodyx"))
    else:
        rows = [["Type", "Asset", "Indication", "Confidence"]]
        for c in tc[:12]:
            rows.append([
                c.get("change_type") or "",
                c.get("asset") or "",
                c.get("indication") or "",
                f"{float(c.get('confidence',0)):.2f}",
            ])
        t = Table(rows, colWidths=[1.2*inch, 2.0*inch, 2.7*inch, 0.6*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("FONTSIZE", (0,0), (-1,-1), 8.5),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(t)

    story.append(Spacer(1, 10))

    # Snapshot counts
    story.append(P("Pipeline snapshot (counts by phase)", "H2x"))
    rows = [["Phase", "Count"]] + [[k, str(v)] for k, v in sorted(counts.items(), key=lambda x: x[0])]
    rows.append(["Total", str(sum(counts.values()))])
    t2 = Table(rows, colWidths=[5.0*inch, 0.9*inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("GRID",(0,0),(-1,-1),0.25,colors.lightgrey),
        ("FONTSIZE",(0,0),(-1,-1),9),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    # Corroboration summary
    story.append(P("Corroboration summary", "H2x"))
    if ctgov_summary.get("enabled"):
        story.append(P(
            f"ClinicalTrials.gov: matches for {ctgov_summary.get('programs_with_matches')}/{ctgov_summary.get('programs_total')} programs; "
            f"phase mismatches: {ctgov_summary.get('phase_mismatches')}.",
            "Bodyx"
        ))
    else:
        story.append(P("ClinicalTrials.gov: disabled.", "Bodyx"))

    ed_sum = (edgar_pack or {}).get("summary") or {}
    if ed_sum.get("enabled"):
        story.append(P(
            f"SEC EDGAR: {ed_sum.get('filings_with_events')} filings with pipeline-related signals since {ed_sum.get('since_date')} "
            f"(new filings considered: {ed_sum.get('filings_considered')}).",
            "Bodyx"
        ))
    else:
        story.append(P("SEC EDGAR: disabled.", "Bodyx"))

    pr_sum = (press_pack or {}).get("summary") or {}
    if pr_sum.get("enabled"):
        story.append(P(
            f"Press releases: {pr_sum.get('releases_with_events')} releases with pipeline-relevant signals (new releases processed: {pr_sum.get('new_releases_processed')}).",
            "Bodyx"
        ))
    else:
        story.append(P("Press releases: disabled.", "Bodyx"))

    story.append(PageBreak())

    # Appendix inventory
    story.append(P("Appendix — Full pipeline inventory", "H2x"))
    story.append(P("Includes source page for manual verification and CT.gov phase flag when available.", "Smallx"))

    inv_rows = [["Asset", "Indication", "Phase", "Page", "CT.gov phase", "Flag"]]
    for p in sorted(programs, key=lambda x: (x.get("phase") or "ZZZ", x.get("asset") or "")):
        inv_rows.append([
            p.get("asset") or "",
            p.get("indication") or "",
            p.get("phase") or "",
            str(p.get("source_page") or ""),
            p.get("_ctgov_best_phase") or "",
            p.get("_ctgov_phase_flag") or "",
        ])
    inv = Table(inv_rows, colWidths=[1.6*inch, 2.6*inch, 0.9*inch, 0.4*inch, 0.9*inch, 0.5*inch], repeatRows=1)
    inv.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("GRID",(0,0),(-1,-1),0.25,colors.lightgrey),
        ("FONTSIZE",(0,0),(-1,-1),7.5),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
    ]))
    story.append(inv)

    doc.build(story)


# -------------------------
# Main
# -------------------------

def main() -> None:
    with open("config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    company = cfg.get("company") or (cfg.get("companies") or [None])[0]
    if not company:
        raise RuntimeError("config.yml must contain 'company' (J&J-only repo)")

    llm_models = cfg.get("llm_models") or [cfg.get("llm_model", "gemini-2.5-flash")]

    name = company.get("name", "Johnson & Johnson")
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

    # EDGAR lookback window (fixes “since today” issue)
    lookback_days = int(company.get("sec_lookback_days", 90))
    since_date = (datetime.datetime.utcnow().date() - datetime.timedelta(days=lookback_days)).isoformat()

    # 1) source URL
    source_type = company.get("source_type") or "jnj_q4cdn_auto"
    if source_type == "jnj_q4cdn_auto":
        source_url = discover_latest_jnj_pipeline_pdf(
            q4cdn_base=company["q4cdn_base"],
            filename_prefix=company.get("filename_prefix", "JNJ-Pipeline"),
            lookback_quarters=int(company.get("lookback_quarters", 12)),
        )
    elif source_type == "direct_pdf":
        source_url = company["pdf_url"]
    else:
        raise RuntimeError("Unsupported source_type for this repo.")

    # 2) download PDF + audit store
    pdf_resp = http_get(source_url)
    pdf_resp.raise_for_status()
    pdf_bytes = pdf_resp.content
    pdf_hash = sha256_bytes(pdf_bytes)

    source_pdf_path = os.path.join(sources_dir, f"{run_date}.pdf")
    with open(source_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    safe_text_dump(os.path.join(raw_dir, f"{run_date}.source.txt"),
                   f"Source URL: {source_url}\nStored PDF: {source_pdf_path}\nSHA256: {pdf_hash}\n")

    # 3) reuse if unchanged
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
        diff = {"added": [], "removed": [], "phase_changes": []}
        safe_text_dump(os.path.join(raw_dir, f"{run_date}.extracted_text_preview.txt"),
                       "Reused prior snapshot (PDF unchanged). No extraction performed.\n")
    else:
        use_vision = bool(company.get("use_vision", True))
        vision_max_pages = int(company.get("vision_max_pages", 10))
        vision_dpi = int(company.get("vision_dpi", 190))

        snapshot = {"as_of_date": None, "programs": []}
        extraction_mode = "none"
        errors: List[str] = []

        if use_vision:
            try:
                extraction_mode = "vision"
                snapshot = llm_extract_pipeline_vision(name, pdf_bytes, llm_models, max_pages=vision_max_pages, dpi=vision_dpi)
            except Exception as e:
                errors.append(f"vision_error: {e}")
                snapshot = {"as_of_date": None, "programs": []}

        if program_count(snapshot) == 0:
            try:
                extraction_mode = "text+ocr"
                extracted_text = pdf_to_text(pdf_bytes)
                combined_text = maybe_add_ocr(pdf_bytes, extracted_text, ocr_max_pages=8, ocr_dpi=220)
                safe_text_dump(os.path.join(raw_dir, f"{run_date}.extracted_text_preview.txt"), combined_text[:25000])
                snapshot = llm_extract_pipeline_text(name, combined_text[:160000], llm_models)
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
            "extraction_ok": program_count(snapshot) > 0,
        })

        diff = diff_programs(prev_snapshot or {}, snapshot)

    # 4) CT.gov corroboration
    ctgov_summary = ctgov_corroborate(snapshot, diff, company, run_date, sources_dir=sources_dir)

    # 5) EDGAR corroboration (improved)
    edgar_pack = {"summary": {"enabled": False}, "filing_events": []}
    if company.get("sec_enabled", True):
        edgar_pack = edgar_corroborate(
            snapshot=snapshot,
            cfg_company=company,
            models=llm_models,
            run_date=run_date,
            sources_dir=sources_dir,
            since_date=since_date,
        )

    # 6) Press release corroboration (new)
    press_pack = {"summary": {"enabled": False}, "press_events": []}
    if company.get("press_enabled", False):
        press_pack = press_corroborate(
            snapshot=snapshot,
            cfg_company=company,
            models=llm_models,
            run_date=run_date,
            sources_dir=sources_dir,
        )

    # 7) Final narrative synthesis
    final_brief = llm_final_brief(name, diff, ctgov_summary, edgar_pack, press_pack, models=llm_models)

    # 8) persist snapshots
    dated_snapshot_path = os.path.join(snap_dir, f"{run_date}.json")
    safe_json_dump(dated_snapshot_path, snapshot)

    latest_ok = bool((snapshot.get("_meta", {}) or {}).get("extraction_ok", False))
    if reused_snapshot or latest_ok:
        safe_json_dump(prev_path, snapshot)

    # 9) outputs
    programs_csv_path = os.path.join(rep_dir, f"{run_date}.programs.csv")
    export_programs_csv(programs_csv_path, snapshot.get("programs") or [])

    evidence_json_path = os.path.join(rep_dir, f"{run_date}.evidence.json")
    safe_json_dump(evidence_json_path, {
        "run_date_utc": run_date,
        "pipeline_source": {"url": source_url, "stored_pdf": source_pdf_path, "sha256": pdf_hash},
        "diff": diff,
        "ctgov": ctgov_summary,
        "sec_edgar": edgar_pack,
        "press": press_pack,
        "final_brief": final_brief,
    })

    md_path = os.path.join(rep_dir, f"{run_date}.md")
    pdf_path = os.path.join(rep_dir, f"{run_date}.pdf")

    md = render_markdown(
        company=company,
        run_date=run_date,
        snapshot=snapshot,
        diff=diff,
        final_brief=final_brief,
        source_url=source_url,
        source_pdf_path=source_pdf_path,
        source_sha256=pdf_hash,
        reused_snapshot=reused_snapshot,
        recovered_baseline=recovered_baseline,
        programs_csv_path=programs_csv_path,
        evidence_json_path=evidence_json_path,
        ctgov_summary=ctgov_summary,
        edgar_pack=edgar_pack,
        press_pack=press_pack,
    )
    safe_text_dump(md_path, md)

    build_pdf(
        pdf_path=pdf_path,
        company=company,
        run_date=run_date,
        snapshot=snapshot,
        diff=diff,
        final_brief=final_brief,
        source_url=source_url,
        source_pdf_path=source_pdf_path,
        source_sha256=pdf_hash,
        reused_snapshot=reused_snapshot,
        recovered_baseline=recovered_baseline,
        programs_csv_path=programs_csv_path,
        evidence_json_path=evidence_json_path,
        ctgov_summary=ctgov_summary,
        edgar_pack=edgar_pack,
        press_pack=press_pack,
    )

    print("Done.")
    print("Source:", source_url)
    print("Stored PDF:", source_pdf_path)
    print("Snapshot:", dated_snapshot_path)
    print("MD report:", md_path)
    print("PDF report:", pdf_path)
    print("Evidence:", evidence_json_path)
    print("CSV:", programs_csv_path)


if __name__ == "__main__":
    main()
