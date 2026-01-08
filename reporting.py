import os
from typing import Dict, List, Tuple, Any

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _phase_norm(p: str) -> str:
    if not p:
        return "Unknown"
    x = str(p).strip().lower()
    # Basic normalization – customize if you want
    if "reg" in x or "file" in x or "submission" in x:
        return "Registration"
    if "3" in x and "phase" in x:
        return "Phase 3"
    if "2" in x and "phase" in x:
        return "Phase 2"
    if "1" in x and "phase" in x:
        return "Phase 1"
    if x in {"p1", "p2", "p3"}:
        return {"p1": "Phase 1", "p2": "Phase 2", "p3": "Phase 3"}[x]
    return p.strip() or "Unknown"


def _count_by_phase(programs: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in programs or []:
        ph = _phase_norm(p.get("phase") or "")
        out[ph] = out.get(ph, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceAfter=10))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceAfter=8))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], leading=14))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=11))
    return styles


def _table(data: List[List[str]], col_widths=None) -> Table:
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ALIGN", (0, 0), (-1, 0), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
    ]))
    return t


def build_snapshot_markdown(
    company_name: str,
    run_date: str,
    snapshot: dict,
    source_url: str,
    source_pdf_path: str,
    source_sha256: str,
    programs_csv_path: str,
    evidence_json_path: str,
    ctgov_summary: dict,
    edgar_pack: dict,
    press_pack: dict,
    reused_snapshot: bool,
    recovered_baseline: bool,
) -> str:
    programs = _safe_list(snapshot.get("programs"))
    counts = _count_by_phase(programs)

    lines: List[str] = []
    lines.append(f"# Snapshot Pipeline Intelligence — {company_name}")
    lines.append("")
    lines.append("## Run coverage & provenance")
    lines.append(f"- Run date (UTC): {run_date}")
    lines.append(f"- Pipeline source URL: {source_url}")
    lines.append(f"- Stored pipeline PDF: `{source_pdf_path}`")
    lines.append(f"- Pipeline SHA256: `{source_sha256}`")
    lines.append(f"- Programs extracted: {len(programs)}")
    if recovered_baseline:
        lines.append("- Note: baseline recovery was used (latest.json invalid).")
    if reused_snapshot:
        lines.append("- Note: source PDF unchanged; reused prior extracted snapshot to avoid OCR/LLM variability.")
    lines.append(f"- Inventory export (CSV): `{programs_csv_path}`")
    lines.append(f"- Evidence pack (JSON): `{evidence_json_path}`")
    lines.append("")

    lines.append("## Pipeline size by phase (snapshot)")
    if not counts:
        lines.append("_No programs available._")
    else:
        for k, v in counts.items():
            lines.append(f"- {k}: {v}")
    lines.append("")

    # CT.gov summary
    lines.append("## ClinicalTrials.gov coverage")
    if not ctgov_summary.get("enabled"):
        lines.append("_Disabled._")
    else:
        lines.append(f"- API base: {ctgov_summary.get('api_base')}")
        lines.append(f"- Programs with matches: {ctgov_summary.get('programs_with_matches')}/{ctgov_summary.get('programs_total')}")
        lines.append(f"- Phase mismatches: {ctgov_summary.get('phase_mismatches')}")
        lines.append(f"- Stored CT.gov evidence: `{ctgov_summary.get('sources_path')}`")
    lines.append("")

    # EDGAR coverage list if present
    lines.append("## SEC EDGAR coverage")
    ed_sum = (edgar_pack or {}).get("summary") or {}
    if not ed_sum.get("enabled"):
        lines.append("_Disabled._")
    else:
        lines.append(f"- Filings since: {ed_sum.get('since_date')}")
        lines.append(f"- New filings processed: {ed_sum.get('filings_considered')}")
        lines.append(f"- Filings with extracted events: {ed_sum.get('filings_with_events')}")
        lines.append(f"- Stored EDGAR evidence: `{ed_sum.get('sources_path')}`")
    lines.append("")

    # Press coverage list if present
    lines.append("## Press coverage")
    pr_sum = (press_pack or {}).get("summary") or {}
    feed_items = _safe_list((press_pack or {}).get("feed_items"))
    if not pr_sum.get("enabled"):
        lines.append("_Disabled._")
    else:
        lines.append(f"- Feed URL: {pr_sum.get('feed_url')}")
        lines.append(f"- Links scanned: {pr_sum.get('links_scanned')}")
        lines.append(f"- New releases processed: {pr_sum.get('new_releases_processed')}")
        lines.append(f"- Releases with extracted events: {pr_sum.get('releases_with_extracted_events') or pr_sum.get('releases_with_events')}")
        lines.append(f"- Stored press evidence: `{pr_sum.get('sources_path')}`")
        lines.append("")
        if feed_items:
            lines.append("### Latest press releases (coverage)")
            for it in feed_items[:10]:
                status = "already processed" if it.get("already_processed") else "new"
                lines.append(f"- {it.get('date') or ''} — {status} — {it.get('title') or ''}")
                lines.append(f"  - {it.get('url')}")
        else:
            lines.append("_No press headlines captured into report object._")
    lines.append("")

    lines.append("## Pipeline inventory (full list)")
    lines.append("| Asset | Indication | Phase | Page |")
    lines.append("|---|---|---|---:|")
    for p in sorted(programs, key=lambda x: ((x.get("asset") or "").lower(), (x.get("indication") or "").lower())):
        lines.append(f"| {p.get('asset','')} | {p.get('indication','') or ''} | {p.get('phase') or ''} | {p.get('source_page') or ''} |")

    lines.append("")
    lines.append("## Notes & limitations")
    lines.append("- This snapshot reflects public-source extraction; verify critical items in the stored source PDF.")
    lines.append("- Phase information can be encoded visually (e.g., color); corroboration is provided via CT.gov where matched.")
    lines.append("")

    return "\n".join(lines)


def build_delta_markdown(
    company_name: str,
    run_date: str,
    diff: dict,
    final_brief: dict,
    source_url: str,
    source_pdf_path: str,
    source_sha256: str,
) -> str:
    added = _safe_list(diff.get("added"))
    removed = _safe_list(diff.get("removed"))
    pcs = _safe_list(diff.get("phase_changes"))

    lines: List[str] = []
    lines.append(f"# Delta Pipeline Change Report — {company_name}")
    lines.append("")
    lines.append("## Run provenance")
    lines.append(f"- Run date (UTC): {run_date}")
    lines.append(f"- Pipeline source URL: {source_url}")
    lines.append(f"- Stored pipeline PDF: `{source_pdf_path}`")
    lines.append(f"- Pipeline SHA256: `{source_sha256}`")
    lines.append("")

    lines.append("## Executive summary")
    headline = final_brief.get("headline") or "Pipeline monitoring update"
    lines.append(f"**{headline}**")
    if final_brief.get("executive_summary"):
        lines.append("")
        lines.append(final_brief["executive_summary"])
    lines.append("")

    lines.append("## Change summary")
    lines.append(f"- New / Added: {len(added)}")
    lines.append(f"- Removed / Discontinued: {len(removed)}")
    lines.append(f"- Phase changes: {len(pcs)}")
    lines.append("")

    lines.append("## Changes (details)")
    if not added and not removed and not pcs:
        lines.append("_No pipeline changes detected versus the last valid snapshot._")
        lines.append("")
        return "\n".join(lines)

    def sec(title: str, items: List[dict]):
        lines.append(f"### {title}")
        if not items:
            lines.append("_None detected._\n")
            return
        for it in items[:200]:
            lines.append(f"- **{it.get('asset','')}** — {it.get('indication','') or ''} — {it.get('phase') or ''} (p.{it.get('source_page') or ''})")
        lines.append("")

    sec("New / Added items", added)
    sec("Removed / Discontinued items", removed)

    lines.append("### Phase changes")
    if not pcs:
        lines.append("_None detected._\n")
    else:
        for it in pcs[:200]:
            lines.append(f"- **{it.get('asset','')}** — {it.get('indication','') or ''} — {it.get('from_phase')} → {it.get('to_phase')} (p.{it.get('source_page') or ''})")
        lines.append("")

    return "\n".join(lines)


def build_snapshot_pdf(pdf_path: str, md_text: str, company_name: str, run_date: str, programs: List[dict], programs_csv_path: str) -> None:
    ensure_dir(os.path.dirname(pdf_path))
    styles = _pdf_styles()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=LETTER,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    story: List[Any] = []
    story.append(Paragraph(f"Snapshot Pipeline Intelligence — {company_name}", styles["Title"]))
    story.append(Paragraph(f"Run date (UTC): {run_date}", styles["Small"]))
    story.append(Spacer(1, 0.2 * inch))

    # Key facts table
    counts = _count_by_phase(programs)
    key_rows = [["Key facts", ""]]
    key_rows.append(["Programs extracted", str(len(programs))])
    key_rows.append(["Inventory export", programs_csv_path])
    if counts:
        key_rows.append(["Phase distribution", ", ".join([f"{k}: {v}" for k, v in counts.items()])])
    story.append(_table(key_rows, col_widths=[2.1 * inch, 4.9 * inch]))
    story.append(Spacer(1, 0.25 * inch))

    # Sections as text (clean, professional)
    for block in md_text.split("\n## "):
        if not block.strip():
            continue
        if block.startswith("# "):
            continue
        if block.startswith("Snapshot Pipeline Intelligence"):
            continue
        if block.startswith("Run coverage"):
            title, body = block.split("\n", 1) if "\n" in block else (block, "")
            story.append(Paragraph("Run coverage & provenance", styles["H1"]))
            story.append(Spacer(1, 0.05 * inch))
            story.append(Paragraph(body.replace("\n", "<br/>"), styles["Small"]))
            story.append(Spacer(1, 0.15 * inch))
            continue

    # Inventory appendix (first N)
    story.append(PageBreak())
    story.append(Paragraph("Appendix — Pipeline inventory (first 60 rows)", styles["H1"]))
    story.append(Paragraph(f"Full inventory in CSV: {programs_csv_path}", styles["Small"]))
    story.append(Spacer(1, 0.15 * inch))

    data = [["Asset", "Indication", "Phase", "Page"]]
    for p in (programs or [])[:60]:
        data.append([
            (p.get("asset") or "")[:60],
            (p.get("indication") or "")[:70],
            (p.get("phase") or "")[:20],
            str(p.get("source_page") or ""),
        ])
    story.append(_table(data, col_widths=[1.7 * inch, 3.2 * inch, 0.9 * inch, 0.7 * inch]))

    doc.build(story)


def build_delta_pdf(pdf_path: str, company_name: str, run_date: str, delta_md: str, diff: dict) -> None:
    ensure_dir(os.path.dirname(pdf_path))
    styles = _pdf_styles()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=LETTER,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    added = _safe_list(diff.get("added"))
    removed = _safe_list(diff.get("removed"))
    pcs = _safe_list(diff.get("phase_changes"))

    story: List[Any] = []
    story.append(Paragraph(f"Delta Pipeline Change Report — {company_name}", styles["Title"]))
    story.append(Paragraph(f"Run date (UTC): {run_date}", styles["Small"]))
    story.append(Spacer(1, 0.2 * inch))

    # Summary table
    rows = [["Change summary", "Count"]]
    rows.append(["New / Added", str(len(added))])
    rows.append(["Removed / Discontinued", str(len(removed))])
    rows.append(["Phase changes", str(len(pcs))])
    story.append(_table(rows, col_widths=[5.0 * inch, 2.0 * inch]))
    story.append(Spacer(1, 0.25 * inch))

    # If no changes, explicitly say so (but still professional)
    if not added and not removed and not pcs:
        story.append(Paragraph("No pipeline changes detected versus the last valid snapshot.", styles["Body"]))
        doc.build(story)
        return

    # Detail tables (top N each)
    def section_table(title: str, items: List[dict], mode: str):
        story.append(Paragraph(title, styles["H2"]))
        if not items:
            story.append(Paragraph("None detected.", styles["Body"]))
            story.append(Spacer(1, 0.1 * inch))
            return
        data = [["Asset", "Indication", "Phase / Transition", "Page"]]
        for it in items[:40]:
            if mode == "phase":
                phase = f"{it.get('from_phase')} → {it.get('to_phase')}"
            else:
                phase = it.get("phase") or ""
            data.append([
                (it.get("asset") or "")[:55],
                (it.get("indication") or "")[:70],
                phase[:25],
                str(it.get("source_page") or ""),
            ])
        story.append(_table(data, col_widths=[1.7 * inch, 3.2 * inch, 1.3 * inch, 0.6 * inch]))
        story.append(Spacer(1, 0.2 * inch))

    section_table("New / Added items", added, mode="std")
    section_table("Removed / Discontinued items", removed, mode="std")
    section_table("Phase changes", pcs, mode="phase")

    doc.build(story)


def write_snapshot_and_delta_reports(
    report_dir: str,
    company_name: str,
    run_date: str,
    snapshot: dict,
    diff: dict,
    final_brief: dict,
    source_url: str,
    source_pdf_path: str,
    source_sha256: str,
    programs_csv_path: str,
    evidence_json_path: str,
    ctgov_summary: dict,
    edgar_pack: dict,
    press_pack: dict,
    reused_snapshot: bool,
    recovered_baseline: bool,
) -> Dict[str, str]:
    ensure_dir(report_dir)

    programs = _safe_list(snapshot.get("programs"))

    # Build snapshot report (always full)
    snapshot_md = build_snapshot_markdown(
        company_name=company_name,
        run_date=run_date,
        snapshot=snapshot,
        source_url=source_url,
        source_pdf_path=source_pdf_path,
        source_sha256=source_sha256,
        programs_csv_path=programs_csv_path,
        evidence_json_path=evidence_json_path,
        ctgov_summary=ctgov_summary,
        edgar_pack=edgar_pack,
        press_pack=press_pack,
        reused_snapshot=reused_snapshot,
        recovered_baseline=recovered_baseline,
    )
    snapshot_md_path = os.path.join(report_dir, f"{run_date}.snapshot.md")
    snapshot_pdf_path = os.path.join(report_dir, f"{run_date}.snapshot.pdf")
    _write_text(snapshot_md_path, snapshot_md)
    build_snapshot_pdf(snapshot_pdf_path, snapshot_md, company_name, run_date, programs, programs_csv_path)

    # Build delta report (can be "no changes", still a clean PDF)
    delta_md = build_delta_markdown(
        company_name=company_name,
        run_date=run_date,
        diff=diff,
        final_brief=final_brief or {},
        source_url=source_url,
        source_pdf_path=source_pdf_path,
        source_sha256=source_sha256,
    )
    delta_md_path = os.path.join(report_dir, f"{run_date}.delta.md")
    delta_pdf_path = os.path.join(report_dir, f"{run_date}.delta.pdf")
    _write_text(delta_md_path, delta_md)
    build_delta_pdf(delta_pdf_path, company_name, run_date, delta_md, diff)

    # Optional backward-compatible outputs (if your workflow expects these names)
    # Keep "YYYY-MM-DD.md/pdf" as the snapshot report.
    legacy_md = os.path.join(report_dir, f"{run_date}.md")
    legacy_pdf = os.path.join(report_dir, f"{run_date}.pdf")
    _write_text(legacy_md, snapshot_md)
    # Copy snapshot pdf to legacy name
    try:
        import shutil
        shutil.copyfile(snapshot_pdf_path, legacy_pdf)
    except Exception:
        pass

    return {
        "snapshot_md": snapshot_md_path,
        "snapshot_pdf": snapshot_pdf_path,
        "delta_md": delta_md_path,
        "delta_pdf": delta_pdf_path,
        "legacy_md": legacy_md,
        "legacy_pdf": legacy_pdf,
    }
