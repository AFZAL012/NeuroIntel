"""
Phase J — Longitudinal MRI Analysis
Track multiple MRI scans over time, compute growth/shrinkage trends.
"""
from __future__ import annotations
import json
import os
import uuid
from typing import Dict, List, Any
from datetime import datetime


SESSIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "static", "sessions"
)
os.makedirs(SESSIONS_DIR, exist_ok=True)


def get_or_create_session(session_id: str | None = None) -> str:
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
    path = _session_path(session_id)
    if not os.path.exists(path):
        _save_session(session_id, {"scans": []})
    return session_id


def add_scan(
    session_id: str,
    scan_date:  str,
    filename:   str,
    tumor_type: str,
    area_pct:   float,
    volume_mm3: float,
    neuroscore: float,
) -> Dict[str, Any]:
    """Add a new scan record to the longitudinal session."""
    data = _load_session(session_id)
    record = {
        "date":       scan_date or datetime.now().strftime("%Y-%m-%d"),
        "filename":   filename,
        "tumor_type": tumor_type,
        "area_pct":   area_pct,
        "volume_mm3": volume_mm3,
        "neuroscore": neuroscore,
    }
    data["scans"].append(record)
    # Keep max 10 scans
    if len(data["scans"]) > 10:
        data["scans"] = data["scans"][-10:]
    _save_session(session_id, data)
    return compute_longitudinal(session_id)


def compute_longitudinal(session_id: str) -> Dict[str, Any]:
    """Compute growth trends from session scan history."""
    data  = _load_session(session_id)
    scans = data.get("scans", [])

    if len(scans) < 2:
        return {
            "session_id": session_id,
            "scans": scans,
            "trend": "insufficient_data",
            "growth_rate": None,
            "shrinkage_rate": None,
            "chart_data": _build_chart_data(scans),
        }

    areas    = [s["area_pct"]   for s in scans]
    volumes  = [s["volume_mm3"] for s in scans]
    scores   = [s["neuroscore"] for s in scans]
    dates    = [s["date"]       for s in scans]

    # Growth between first and last
    first_area, last_area = areas[0], areas[-1]
    if first_area > 0:
        pct_change = (last_area - first_area) / first_area * 100
    else:
        pct_change = 0.0

    growth_rate   = round(pct_change, 2) if pct_change > 0  else 0.0
    shrinkage_rate = round(abs(pct_change), 2) if pct_change < 0 else 0.0

    # Trend classification
    if abs(pct_change) < 5:
        trend = "stable"
    elif pct_change > 0:
        trend = "growing"
    else:
        trend = "shrinking"

    # Volume change
    first_v, last_v = volumes[0], volumes[-1]
    vol_change_pct = round((last_v - first_v) / (first_v + 1e-6) * 100, 1)

    # Progression score (NeuroScore delta)
    first_ns, last_ns = scores[0], scores[-1]
    ns_delta = round(last_ns - first_ns, 1)

    return {
        "session_id":     session_id,
        "scans":          scans,
        "n_scans":        len(scans),
        "trend":          trend,
        "growth_rate":    growth_rate,
        "shrinkage_rate": shrinkage_rate,
        "pct_change":     round(pct_change, 2),
        "vol_change_pct": vol_change_pct,
        "ns_delta":       ns_delta,
        "chart_data":     _build_chart_data(scans),
        "clinical_note":  _longitudinal_note(trend, pct_change, ns_delta),
    }


def clear_session(session_id: str) -> None:
    _save_session(session_id, {"scans": []})


def _build_chart_data(scans: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "labels":    [s["date"]       for s in scans],
        "area_pct":  [s["area_pct"]   for s in scans],
        "volume":    [s["volume_mm3"] for s in scans],
        "neuroscore":[s["neuroscore"] for s in scans],
    }


def _longitudinal_note(trend, pct_change, ns_delta) -> str:
    if trend == "insufficient_data":
        return "Upload at least 2 scans to enable longitudinal analysis."
    if trend == "stable":
        return ("Tumour volume has remained stable across scans (< 5% change). "
                "Continue current monitoring protocol.")
    if trend == "growing":
        return (f"Tumour has grown by {abs(pct_change):.1f}% since baseline. "
                f"NeuroScore delta: {ns_delta:+.1f}. "
                "Escalation of treatment plan recommended. Urgent specialist review.")
    return (f"Tumour has shrunk by {abs(pct_change):.1f}% since baseline. "
            f"NeuroScore delta: {ns_delta:+.1f}. "
            "Treatment response appears positive. Continue and monitor.")


def _session_path(sid: str) -> str:
    return os.path.join(SESSIONS_DIR, f"session_{sid}.json")


def _load_session(sid: str) -> Dict[str, Any]:
    path = _session_path(sid)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"scans": []}


def _save_session(sid: str, data: Dict[str, Any]) -> None:
    with open(_session_path(sid), "w") as f:
        json.dump(data, f, indent=2)
