"""Portfolio peer filtering and cross-report field benchmarking."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
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


@dataclass(frozen=True)
class BenchmarkObservation:
    """A value normalized enough to decide whether two reports are comparable."""

    file: str
    value: Decimal
    unit: str
    currency: Optional[str] = None
    commodity: Optional[str] = None
    raw: str = ""

    @property
    def comparison_key(self) -> tuple[str, Optional[str], Optional[str]]:
        return (self.unit, self.currency, self.commodity)


_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")
_PERCENT_RE = re.compile(
    r"([-+]?(?:\d[\d,]*\.?\d*|\.\d+))\s*%\s*([A-Za-z]{1,4})?",
    flags=re.IGNORECASE,
)


def _decimal(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    match = _NUMBER_RE.search(str(value))
    if not match:
        return None
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None


def _currency(text: str, fallback: Any = None) -> Optional[str]:
    upper = text.upper().replace(" ", "")
    for currency, tokens in (
        ("CAD", ("CAD", "C$")),
        ("AUD", ("AUD", "A$")),
        ("USD", ("USD", "US$")),
        ("EUR", ("EUR", "€")),
        ("GBP", ("GBP", "£")),
    ):
        if any(token in upper for token in tokens):
            return currency
    if "$" in text:
        return "USD"
    normalized = str(fallback or "").strip().upper()
    return normalized or None


def _money_multiplier(text: str) -> Decimal:
    normalized = text.lower()
    if re.search(r"\b(billion|bn)\b", normalized) or re.search(
        r"(?:[$€£]|\b[A-Z]{3}\s*)\s*\d[\d,.]*\s*b\b",
        text,
        flags=re.IGNORECASE,
    ):
        return Decimal("1000000000")
    if re.search(r"\b(million|mm)\b", normalized) or re.search(
        r"(?:[$€£]|\b[A-Z]{3}\s*)\s*\d[\d,.]*\s*m\b",
        text,
        flags=re.IGNORECASE,
    ):
        return Decimal("1000000")
    if re.search(r"\b(thousand|k)\b", normalized):
        return Decimal("1000")
    return Decimal(1)


def _commodity(value: Any) -> Optional[str]:
    normalized = str(value or "").strip().upper()
    return normalized or None


def _cutoff_unit(text: str, legacy_unit: Any = None) -> str:
    normalized = text.lower().replace(" ", "")
    legacy = str(legacy_unit or "").strip()
    if "%" in text or legacy == "%":
        return "%"
    for pattern, unit in (
        (r"g/t|gpt|g/tonne", "g/t"),
        (r"kg/t|kgpt|kg/tonne", "kg/t"),
        (r"oz/t|opt|oz/ton", "oz/t"),
        (r"lb/t|lb/ton", "lb/t"),
        (r"ppm", "ppm"),
        (r"ppb", "ppb"),
    ):
        if re.search(pattern, normalized) or re.search(
            pattern, legacy.lower().replace(" ", "")
        ):
            return unit
    return legacy.lower() or "unknown"


def _money_observation(
    *,
    filename: str,
    value: Any,
    unit: str,
    currency_hint: Any = None,
) -> Optional[BenchmarkObservation]:
    number = _decimal(value)
    if number is None:
        return None
    raw = str(value)
    denominator = ""
    if unit == "money/rate":
        lowered = raw.lower()
        for pattern, label in (
            (r"/\s*(?:t|tonne)\b|per\s+(?:t|tonne)\b", "t"),
            (r"/\s*lb\b|per\s+lb\b", "lb"),
            (r"/\s*oz\b|per\s+oz\b", "oz"),
            (r"/\s*kg\b|per\s+kg\b", "kg"),
        ):
            if re.search(pattern, lowered):
                denominator = f"/{label}"
                break
        if not denominator:
            denominator = "/unknown"
    return BenchmarkObservation(
        file=filename,
        value=number * _money_multiplier(raw),
        unit=("money" if unit == "money" else f"money{denominator}"),
        currency=_currency(raw, currency_hint),
        raw=raw,
    )


def _percent_observations(
    filename: str,
    value: Any,
    *,
    commodity_hint: Any = None,
) -> List[BenchmarkObservation]:
    if value is None:
        return []
    raw = str(value)
    matches = list(_PERCENT_RE.finditer(raw))
    if matches:
        return [
            BenchmarkObservation(
                file=filename,
                value=Decimal(match.group(1).replace(",", "")),
                unit="%",
                commodity=_commodity(match.group(2) or commodity_hint),
                raw=raw,
            )
            for match in matches
        ]
    number = _decimal(value)
    if number is None:
        return []
    return [
        BenchmarkObservation(
            file=filename,
            value=number,
            unit="%",
            commodity=_commodity(commodity_hint),
            raw=raw,
        )
    ]


def _extract_numeric_values(
    report: dict,
    field: str,
) -> List[BenchmarkObservation]:
    """Normalize current-schema and legacy benchmark values from one report."""
    out: List[BenchmarkObservation] = []
    name = report.get("source_file") or "unknown"

    if field in ("cutoff", "cutoff_grade", "resource_cutoff"):
        rows = list(report.get("mineral_resources") or []) + list(
            report.get("mineral_reserves") or []
        )
        for row in rows:
            value = row.get("cut_off_grade", row.get("cutoff_grade"))
            number = _decimal(value)
            if number is None:
                continue
            raw = str(value)
            out.append(
                BenchmarkObservation(
                    file=name,
                    value=number,
                    unit=_cutoff_unit(raw, row.get("cutoff_unit")),
                    commodity=_commodity(row.get("commodity")),
                    raw=raw,
                )
            )

    elif field in ("npv", "post_tax_npv"):
        econ = report.get("economics") or {}
        for key in ("post_tax_npv", "pre_tax_npv"):
            observation = _money_observation(
                filename=name,
                value=econ.get(key),
                unit="money",
                currency_hint=econ.get("npv_currency"),
            )
            if observation:
                out.append(observation)
                break

    elif field == "irr":
        out.extend(
            _percent_observations(
                name,
                (report.get("economics") or {}).get("irr"),
            )
        )

    elif field in ("initial_capex", "capex"):
        econ = report.get("economics") or {}
        observation = _money_observation(
            filename=name,
            value=econ.get("initial_capex") or econ.get("total_capex"),
            unit="money",
            currency_hint=econ.get("capex_currency"),
        )
        if observation:
            out.append(observation)

    elif field == "opex":
        econ = report.get("economics") or {}
        observation = _money_observation(
            filename=name,
            value=econ.get("opex"),
            unit="money/rate",
            currency_hint=econ.get("opex_currency"),
        )
        if observation:
            out.append(observation)

    elif field in ("recovery", "recovery_rate"):
        econ = report.get("economics") or {}
        out.extend(_percent_observations(name, econ.get("recovery_rate")))
        processing = report.get("processing_method") or {}
        for recovery in processing.get("recoveries") or []:
            out.extend(_percent_observations(name, recovery))
        out.extend(_percent_observations(name, processing.get("overall_recovery")))

    elif field == "dilution":
        mining = report.get("mining_method") or {}
        out.extend(
            _percent_observations(
                name,
                mining.get("dilution", mining.get("dilution_pct")),
            )
        )

    elif field == "mining_recovery":
        mining = report.get("mining_method") or {}
        out.extend(
            _percent_observations(
                name,
                mining.get("mine_recovery", mining.get("mining_recovery_pct")),
            )
        )

    return out


def benchmark_field(
    field: str,
    peer_reports: List[dict],
    target_report: Optional[dict] = None,
) -> Dict[str, Any]:
    """Aggregate only values that share a unit, currency, and commodity."""
    peer_observations = [
        observation
        for report in peer_reports
        for observation in _extract_numeric_values(report, field)
    ]
    target_observations = (
        _extract_numeric_values(target_report, field) if target_report else []
    )

    grouped: dict[
        tuple[str, Optional[str], Optional[str]],
        List[BenchmarkObservation],
    ] = {}
    for observation in peer_observations:
        grouped.setdefault(observation.comparison_key, []).append(observation)

    if target_observations:
        target_keys = {
            observation.comparison_key for observation in target_observations
        }
        comparison_key = min(
            target_keys,
            key=lambda key: (-len(grouped.get(key, [])), repr(key)),
        )
    elif grouped:
        comparison_key = min(
            grouped,
            key=lambda key: (-len(grouped[key]), repr(key)),
        )
    else:
        comparison_key = None

    selected_peers = grouped.get(comparison_key, []) if comparison_key else []
    selected_targets = (
        [
            observation
            for observation in target_observations
            if observation.comparison_key == comparison_key
        ]
        if comparison_key
        else target_observations
    )
    peer_values = [observation.value for observation in selected_peers]
    target_values = [observation.value for observation in selected_targets]
    values = peer_values or target_values

    def entry(observation: BenchmarkObservation, *, is_target: bool = False) -> dict:
        payload = {
            "file": observation.file,
            "value": float(observation.value),
            "unit": observation.unit,
            "currency": observation.currency,
            "commodity": observation.commodity,
            "raw": observation.raw,
            "comparable": observation.comparison_key == comparison_key,
        }
        if is_target:
            payload["is_target"] = True
        return payload

    entries = [entry(observation) for observation in peer_observations]
    entries.extend(
        entry(observation, is_target=True)
        for observation in target_observations
    )

    if not values:
        return {
            "field": field,
            "count": 0,
            "summary": f"No '{field}' values found in peer extractions.",
            "entries": entries,
            "outliers": [],
            "comparison_key": None,
            "excluded_incomparable": 0,
        }

    med = statistics.median(values)
    mn, mx = min(values), max(values)

    target_outliers: List[str] = []
    if target_report and target_values and peer_values:
        peer_mn, peer_mx = min(peer_values), max(peer_values)
        peer_med = statistics.median(peer_values)
        for target_value in target_values:
            spread = peer_mx - peer_mn
            if spread > 0 and (
                target_value < peer_mn - Decimal("0.25") * spread
                or target_value > peer_mx + Decimal("0.25") * spread
            ):
                target_outliers.append(
                    f"Target value {float(target_value):.4g} is outside peer range "
                    f"[{float(peer_mn):.4g}, {float(peer_mx):.4g}] "
                    f"(peer median {peer_med:.4g})"
                )
            elif spread == 0 and target_value != peer_mn:
                target_outliers.append(
                    f"Target value {float(target_value):.4g} differs from peer value "
                    f"{float(peer_mn):.4g}"
                )

    summary = (
        f"Field '{field}': peer_n={len(peer_values)}, "
        f"peer_range=[{float(mn):.4g}, {float(mx):.4g}], "
        f"peer_median={float(med):.4g}. "
        + "; ".join(
            f"{observation.file}: {float(observation.value):.4g}"
            for observation in selected_peers[:8]
        )
    )
    if len(selected_peers) > 8:
        summary += f" (+{len(selected_peers) - 8} more)"

    key_payload = (
        {
            "unit": comparison_key[0],
            "currency": comparison_key[1],
            "commodity": comparison_key[2],
        }
        if comparison_key
        else None
    )

    return {
        "field": field,
        "count": len(peer_values) if peer_values else len(target_values),
        "min": float(mn),
        "max": float(mx),
        "median": float(med),
        "summary": summary,
        "entries": entries,
        "outliers": target_outliers,
        "comparison_key": key_payload,
        "excluded_incomparable": len(peer_observations) - len(selected_peers),
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
    if "dilution" in q:
        return "dilution"
    if "mining recovery" in q:
        return "mining_recovery"
    if "recovery" in q:
        return "recovery_rate"
    return None
