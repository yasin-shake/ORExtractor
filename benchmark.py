"""Portfolio peer filtering and cross-report field benchmarking."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

from extractor import list_extractions, load_extraction
from rag_app import Settings


def _coerce_metadata_str(value: Any) -> Optional[str]:
    """Normalize extraction metadata that may be a plain string or nested dict."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, dict):
        for key in ("method", "name", "text", "value", "type", "deposit_type", "commodity"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
        return None
    if isinstance(value, (int, float, bool)):
        s = str(value).strip()
        return s or None
    return None


def _norm(s: Any) -> str:
    return (_coerce_metadata_str(s) or "").lower()


def _report_commodity(report: dict) -> str:
    if report.get("primary_commodity"):
        return _norm(report["primary_commodity"])
    pi = report.get("property_info") or {}
    comms = pi.get("commodities") or []
    if comms:
        return _norm(comms[0] if isinstance(comms[0], str) else str(comms[0]))
    resources = report.get("mineral_resources") or []
    if resources:
        return _norm(resources[0].get("commodity"))
    return ""


def _report_country(report: dict) -> str:
    pi = report.get("property_info") or {}
    return _norm(pi.get("country"))


def _report_deposit_type(report: dict) -> str:
    if report.get("deposit_type"):
        return _norm(report["deposit_type"])
    geo = report.get("geology") or {}
    return _norm(geo.get("deposit_type"))


def _report_study_stage(report: dict) -> str:
    if report.get("study_stage"):
        return _norm(report["study_stage"])
    pi = report.get("property_info") or {}
    return _norm(pi.get("project_stage"))


def _report_mining_method(report: dict) -> str:
    top = report.get("mining_method")
    if isinstance(top, str) and top.strip():
        return _norm(top)
    if isinstance(top, dict):
        return _norm(top.get("method"))
    return ""


def _score_peer(target: dict, candidate: dict) -> int:
    score = 0
    if _report_commodity(target) and _report_commodity(target) == _report_commodity(candidate):
        score += 4
    if _report_deposit_type(target) and _report_deposit_type(target) == _report_deposit_type(candidate):
        score += 3
    if _report_country(target) and _report_country(target) == _report_country(candidate):
        score += 2
    if _report_study_stage(target) and _report_study_stage(target) == _report_study_stage(candidate):
        score += 1
    if _report_mining_method(target) and _report_mining_method(target) == _report_mining_method(candidate):
        score += 1
    return score


def find_peer_reports(
    settings: Settings,
    target_filename: Optional[str] = None,
    commodity: Optional[str] = None,
    country: Optional[str] = None,
    deposit_type: Optional[str] = None,
    mining_method: Optional[str] = None,
    study_stage: Optional[str] = None,
    limit: int = 8,
) -> List[dict]:
    """Return peer report extractions ranked by metadata similarity."""
    all_reports = list_extractions(settings)
    if not all_reports:
        return []

    target: Optional[dict] = None
    if target_filename:
        target = load_extraction(settings, target_filename)
        if target is None:
            target = next(
                (r for r in all_reports if _norm(r.get("source_file")) == _norm(target_filename)),
                None,
            )

    candidates = [r for r in all_reports if not target or r.get("source_file") != target.get("source_file")]

    def _matches_filters(r: dict) -> bool:
        if commodity and commodity.lower() not in _report_commodity(r):
            return False
        if country and country.lower() not in _report_country(r):
            return False
        if deposit_type and deposit_type.lower() not in _report_deposit_type(r):
            return False
        if mining_method and mining_method.lower() not in _report_mining_method(r):
            return False
        if study_stage and study_stage.lower() not in _report_study_stage(r):
            return False
        return True

    filtered = [r for r in candidates if _matches_filters(r)]
    if target:
        filtered.sort(key=lambda r: _score_peer(target, r), reverse=True)
    elif commodity or country or deposit_type:
        filtered.sort(key=lambda r: r.get("source_file", ""))
    else:
        filtered = candidates

    if target and len(filtered) < 3:
        relaxed = [r for r in candidates if r not in filtered]
        relaxed.sort(key=lambda r: _score_peer(target, r), reverse=True)
        for r in relaxed:
            if r not in filtered:
                filtered.append(r)
            if len(filtered) >= limit:
                break

    return filtered[:limit]


def _extract_numeric_values(report: dict, field: str) -> List[Tuple[str, float, str]]:
    """Pull numeric values for a benchmark field from one report."""
    out: List[Tuple[str, float, str]] = []
    name = report.get("source_file") or "unknown"

    if field in ("cutoff", "cutoff_grade", "resource_cutoff"):
        for res in report.get("mineral_resources") or []:
            v = res.get("cutoff_grade")
            if v is not None:
                try:
                    out.append((name, float(v), res.get("cutoff_unit") or ""))
                except (TypeError, ValueError):
                    pass
        for res in report.get("mineral_reserves") or []:
            v = res.get("cutoff_grade")
            if v is not None:
                try:
                    out.append((name, float(v), res.get("cutoff_unit") or ""))
                except (TypeError, ValueError):
                    pass

    elif field in ("npv", "post_tax_npv"):
        econ = report.get("economics") or {}
        for key in ("post_tax_npv", "pre_tax_npv"):
            v = econ.get(key)
            if v is not None:
                try:
                    out.append((name, float(v), econ.get("npv_currency") or "USD"))
                    break
                except (TypeError, ValueError):
                    pass

    elif field in ("irr",):
        econ = report.get("economics") or {}
        v = econ.get("irr")
        if v is not None:
            try:
                out.append((name, float(v), "%"))
            except (TypeError, ValueError):
                pass

    elif field in ("initial_capex", "capex"):
        econ = report.get("economics") or {}
        v = econ.get("initial_capex") or econ.get("total_capex")
        if v is not None:
            try:
                out.append((name, float(v), "USD"))
            except (TypeError, ValueError):
                pass

    elif field in ("opex",):
        econ = report.get("economics") or {}
        v = econ.get("opex")
        if v is not None:
            try:
                out.append((name, float(v), "USD"))
            except (TypeError, ValueError):
                pass

    elif field in ("recovery", "recovery_rate"):
        econ = report.get("economics") or {}
        v = econ.get("recovery_rate")
        if v is not None:
            try:
                out.append((name, float(v), "%"))
            except (TypeError, ValueError):
                pass
        proc = report.get("processing_method") or {}
        v2 = proc.get("overall_recovery") if isinstance(proc, dict) else None
        if v2 is not None:
            try:
                out.append((name, float(v2), "%"))
            except (TypeError, ValueError):
                pass

    elif field in ("dilution",):
        mm = report.get("mining_method") or {}
        if isinstance(mm, dict):
            v = mm.get("dilution_pct")
            if v is not None:
                try:
                    out.append((name, float(v), "%"))
                except (TypeError, ValueError):
                    pass

    elif field in ("mining_recovery",):
        mm = report.get("mining_method") or {}
        if isinstance(mm, dict):
            v = mm.get("mining_recovery_pct")
            if v is not None:
                try:
                    out.append((name, float(v), "%"))
                except (TypeError, ValueError):
                    pass

    return out


def benchmark_field(
    field: str,
    peer_reports: List[dict],
    target_report: Optional[dict] = None,
) -> Dict[str, Any]:
    """Aggregate a numeric field across peers; flag target outliers."""
    peer_values: List[float] = []
    entries: List[dict] = []

    for report in peer_reports:
        for fname, val, unit in _extract_numeric_values(report, field):
            peer_values.append(val)
            entries.append({"file": fname, "value": val, "unit": unit})

    target_vals: List[float] = []
    if target_report:
        for fname, val, unit in _extract_numeric_values(target_report, field):
            target_vals.append(val)
            entries.append({"file": fname, "value": val, "unit": unit, "is_target": True})

    values = peer_values + [v for v in target_vals if v not in peer_values]

    if not values and not peer_values:
        return {
            "field": field,
            "count": 0,
            "summary": f"No '{field}' values found in peer extractions.",
            "entries": entries,
            "outliers": [],
        }

    med = statistics.median(peer_values if peer_values else values)
    try:
        mn, mx = (min(peer_values), max(peer_values)) if peer_values else (min(values), max(values))
    except ValueError:
        mn = mx = med

    target_outliers: List[str] = []
    if target_report and target_vals and peer_values:
        peer_mn, peer_mx = min(peer_values), max(peer_values)
        peer_med = statistics.median(peer_values)
        for tv in target_vals:
            spread = peer_mx - peer_mn
            if spread > 0 and (tv < peer_mn - 0.25 * spread or tv > peer_mx + 0.25 * spread):
                target_outliers.append(
                    f"Target value {tv} is outside peer range [{peer_mn:.4g}, {peer_mx:.4g}] "
                    f"(peer median {peer_med:.4g})"
                )
            elif spread == 0 and tv != peer_mn:
                target_outliers.append(
                    f"Target value {tv} differs from peer value {peer_mn:.4g}"
                )
    elif target_report and target_vals and len(peer_values) == 1:
        if target_vals[0] != peer_values[0]:
            target_outliers.append(
                f"Target value {target_vals[0]} differs from sole peer value {peer_values[0]}"
            )

    summary = (
        f"Field '{field}': peer_n={len(peer_values)}, peer_range=[{mn:.4g}, {mx:.4g}], peer_median={med:.4g}. "
        + "; ".join(f"{e['file']}: {e['value']}" for e in entries[:8])
    )
    if len(entries) > 8:
        summary += f" (+{len(entries) - 8} more)"

    return {
        "field": field,
        "count": len(peer_values) if peer_values else len(values),
        "min": mn,
        "max": mx,
        "median": med,
        "summary": summary,
        "entries": entries,
        "outliers": target_outliers,
    }


def infer_benchmark_field(question: str) -> Optional[str]:
    q = question.lower()
    if "cut-off" in q or "cutoff" in q or "cut off" in q:
        return "cutoff_grade"
    if "npv" in q:
        return "post_tax_npv"
    if "irr" in q:
        return "irr"
    if "capex" in q or "capital cost" in q:
        return "initial_capex"
    if "opex" in q or "operating cost" in q:
        return "opex"
    if "recovery" in q:
        return "recovery_rate"
    if "dilution" in q:
        return "dilution"
    if "mining recovery" in q:
        return "mining_recovery"
    return None
