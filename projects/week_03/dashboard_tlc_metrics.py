#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit dashboard for TLC streaming metrics (Kafka/Redpanda)
--------------------------------------------------------------

Reads compacted metrics from a Kafka topic (default: tlc.metrics) that are
produced by the `tlc_metrics_kafka.py` aggregator.

Features
- Live autorefresh with pause/resume
- Sidebar config (brokers, topic, group, offset start)
- Summary KPIs and per-window charts
- Top PU LocationIDs (Misra–Gries candidates) with time span filter
- Inspect raw metric records, export as JSON/CSV

Requirements
    pip install streamlit confluent-kafka pandas altair

Run
    streamlit run dashboard_tlc_metrics.py

Expected record shape (example)
{
    "ts": "2025-10-10T13:59:53.040670+00:00",
    "window_min": 1,
    "start": "2025-07-03T08:14:00+00:00",
    "end":   "2025-07-03T08:15:00+00:00",
    "events": 14,
    "hll": {"vendors": 2, "od_pairs": 1},
    "mg_top5": [["236", 5], ["140", 2], ...],
    "reservoir_size": 14
}

Note: The metrics topic should be log-compacted (keyed by "{window_min}:{start}").
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st
from confluent_kafka import Consumer, KafkaError

# -----------------------------
# Helpers
# -----------------------------

def _as_dt(x: str) -> datetime:
    try:
        return pd.to_datetime(x, utc=True).to_pydatetime()
    except Exception:
        return datetime.fromisoformat(x.replace("Z", "+00:00"))


def _build_consumer(brokers: str, group: str, topic: str, start_from_end: bool) -> Consumer:
    conf = {
        "bootstrap.servers": brokers,
        "group.id": group,
        "enable.auto.commit": True,
        "auto.offset.reset": "latest" if start_from_end else "earliest",
        "allow.auto.create.topics": True,
    }
    c = Consumer(conf)
    c.subscribe([topic])
    return c


@dataclass
class DashConfig:
    brokers: str
    topic: str
    group: str
    start_from_end: bool
    refresh_ms: int
    max_retained: int  # max windows to keep in memory per window size


# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(page_title="TLC Streaming Metrics", layout="wide")
st.title("🚕 TLC Streaming Metrics Dashboard")

with st.sidebar:
    st.header("⚙️ Settings")
    brokers = st.text_input("Brokers", "localhost:19092")
    topic = st.text_input("Metrics Topic", "tlc.metrics")
    group = st.text_input("Consumer Group", "tlc-metrics-dashboard")
    start_from_end = st.toggle("Start from end (latest)", value=True)
    refresh_ms = st.slider("Auto-refresh (ms)", min_value=200, max_value=5000, value=1000, step=100)
    max_retained = st.slider("Max windows kept (per size)", 50, 5000, 500)

    st.divider()
    st.caption("Use compacted metrics topic for best results.")

cfg = DashConfig(brokers, topic, group, start_from_end, refresh_ms, max_retained)

# Live toggle and manual refresh
cols = st.columns([1,1,3])
with cols[0]:
    if "live" not in st.session_state:
        st.session_state.live = True
    live = st.toggle("Live", value=st.session_state.live, help="Pause to inspect data without consuming")
    st.session_state.live = live
with cols[1]:
    if st.button("↻ Manual refresh"):
        st.toast("Refreshed", icon="🔄")

# Cross-version rerun helper (Streamlit changed experimental_rerun -> rerun)

def _maybe_rerun():
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn and st.session_state.live:
        time.sleep(cfg.refresh_ms / 1000.0)
        fn()

# Cached consumer per settings
@st.cache_resource(show_spinner=False)
def get_consumer(brokers: str, group: str, topic: str, start_from_end: bool) -> Consumer:
    return _build_consumer(brokers, group, topic, start_from_end)

# Buffer for records keyed by "{window_min}:{start}"
if "records" not in st.session_state:
    st.session_state.records = {}

consumer = get_consumer(cfg.brokers, cfg.group, cfg.topic, cfg.start_from_end)

# Poll loop (bounded by refresh budget)
end_by = time.time() + (cfg.refresh_ms / 1000.0) * (0.75 if st.session_state.live else 0.0)
while st.session_state.live and time.time() < end_by:
    msg = consumer.poll(0.01)
    if msg is None:
        continue
    if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
            continue
        st.warning(f"Kafka error: {msg.error()}")
        continue
    try:
        rec = json.loads(msg.value())
        key = msg.key().decode() if msg.key() else f"{rec.get('window_min')}:{rec.get('start')}"
        st.session_state.records[key] = rec
    except Exception as e:
        st.warning(f"Bad record: {e}")

records = list(st.session_state.records.values())
if not records:
    st.info("Waiting for metrics… Ensure aggregator is running and producing to the metrics topic.")
    st.stop()

# Prepare DataFrame
rows = []
for r in records:
    r = dict(r)
    r["start_dt"] = _as_dt(r["start"])                 # window start (data time)
    r["ts_dt"] = _as_dt(r.get("ts", r["start"]))       # publish time (now)
    rows.append(r)

base_df = pd.DataFrame(rows)
base_df.sort_values(["window_min", "start_dt"], inplace=True)

# Sidebar filters
with st.sidebar:
    st.subheader("Filters")
    available_windows = sorted(base_df["window_min"].unique().tolist())
    win_sel = st.multiselect("Window sizes (min)", available_windows, default=available_windows)
    max_hours = st.slider("Show last N hours (by ts)", 1, 24, 6)

min_dt = datetime.now(timezone.utc) - timedelta(hours=max_hours)
view_df = base_df[base_df["window_min"].isin(win_sel)]
view_df = view_df[view_df["ts_dt"] >= min_dt]

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Windows shown", len(view_df))
with col2:
    st.metric("Total events", int(view_df["events"].sum()) if not view_df.empty else 0)
with col3:
    vendors_max = 0
    if not view_df.empty and view_df["hll"].notna().any():
        try:
            vendors_max = int(pd.json_normalize(view_df["hll"]).vendors.max())
        except Exception:
            vendors_max = 0
    st.metric("Max distinct vendors", vendors_max)
with col4:
    od_max = 0
    if not view_df.empty and view_df["hll"].notna().any():
        try:
            od_max = int(pd.json_normalize(view_df["hll"]).od_pairs.max())
        except Exception:
            od_max = 0
    st.metric("Max distinct OD pairs", od_max)

# -----------------------------
# Activity over time (drop nested cols before Altair)
# -----------------------------

st.subheader("Activity over time")
for win in sorted(view_df["window_min"].unique().tolist()):
    sub = view_df[view_df["window_min"] == win].copy()
    if sub.empty:
        continue

    # Flatten HLL
    try:
        hll_df = pd.json_normalize(sub["hll"]).add_prefix("hll.")
        sub = pd.concat([sub.reset_index(drop=True), hll_df.reset_index(drop=True)], axis=1)
    except Exception:
        pass

    # Select only plotting columns to avoid Arrow conversion issues with nested types
    plot_events = sub[["start_dt", "start", "end", "events"]].copy()
    plot_distincts = sub[[c for c in ["start_dt", "start", "end", "hll.vendors", "hll.od_pairs"] if c in sub.columns]].copy()

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Window {win} min** — events")
        chart = alt.Chart(plot_events).mark_line(point=True).encode(
            x=alt.X("start_dt:T", title="Window start"),
            y=alt.Y("events:Q", title="Events"),
            tooltip=["start", "end", "events"],
        ).properties(height=240)
        st.altair_chart(chart, use_container_width=True)

    with right:
        st.markdown(f"**Window {win} min** — distincts")
        if set(["hll.vendors", "hll.od_pairs"]).issubset(plot_distincts.columns):
            chart2 = alt.Chart(plot_distincts).transform_fold(
                ["hll.vendors", "hll.od_pairs"], as_=["metric", "value"]
            ).mark_line(point=True).encode(
                x=alt.X("start_dt:T", title="Window start"),
                y=alt.Y("value:Q", title="Count"),
                color=alt.Color("metric:N", title="Metric"),
# стало (вказуємо типи для полів із transform_fold)
                tooltip=[alt.Tooltip("start:T"), alt.Tooltip("end:T"), alt.Tooltip("metric:N"), alt.Tooltip("value:Q")],
            ).properties(height=240)
            st.altair_chart(chart2, use_container_width=True)
        else:
            st.info("No HLL fields yet for this window size.")

# -----------------------------
# Top PU LocationIDs (Misra–Gries candidates)
# -----------------------------

st.subheader("Top PU LocationIDs (Misra–Gries candidates)")
mg_rows: List[Tuple[str, int]] = []
for _, r in view_df.iterrows():
    try:
        for key, cnt in r["mg_top5"]:
            mg_rows.append((str(key), int(cnt)))
    except Exception:
        pass

if mg_rows:
    mg_df = pd.DataFrame(mg_rows, columns=["pulocation_id", "count"])
    agg_df = mg_df.groupby("pulocation_id", as_index=False)["count"].sum().sort_values("count", ascending=False).head(20)
    bar = alt.Chart(agg_df).mark_bar().encode(
        x=alt.X("count:Q", title="Estimated count (sum of candidates)"),
        y=alt.Y("pulocation_id:N", sort='-x', title="PU Location ID"),
        tooltip=["pulocation_id", "count"],
    ).properties(height=400)
    st.altair_chart(bar, use_container_width=True)
else:
    st.info("No Misra–Gries candidates in the current view.")

# -----------------------------
# Latest windows snapshot
# -----------------------------

st.subheader("Latest windows snapshot")
latest_by_win = view_df.sort_values("start_dt").groupby("window_min").tail(1)
if not latest_by_win.empty:
    tbl = latest_by_win[["window_min", "start", "end", "events"]].copy()
    try:
        hll_last = pd.json_normalize(latest_by_win["hll"]).reset_index(drop=True)
        tbl["vendors≈"] = hll_last.get("vendors", pd.Series([None]*len(tbl))).values
        tbl["od_pairs≈"] = hll_last.get("od_pairs", pd.Series([None]*len(tbl))).values
    except Exception:
        pass
    st.dataframe(tbl, use_container_width=True)
else:
    st.info("No latest windows to display.")

# -----------------------------
# Raw data & export (stringify nested cols to avoid Arrow issues)
# -----------------------------

with st.expander("Raw metric records"):
    to_show = view_df.copy()
    for col in ["hll", "mg_top5"]:
        if col in to_show.columns:
            to_show[col] = to_show[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
    to_show = to_show.drop(columns=["start_dt", "ts_dt"], errors="ignore")
    st.dataframe(to_show, use_container_width=True)

    colx, coly = st.columns([1,1])
    with colx:
        buf = io.StringIO()
        json.dump(to_show.to_dict(orient="records"), buf, ensure_ascii=False)
        st.download_button("Download JSON", data=buf.getvalue(), file_name="metrics.json", mime="application/json")
    with coly:
        csv = to_show.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="metrics.csv", mime="text/csv")

st.caption("© TLC Streaming Metrics — live view. Adjust retention and filters in the sidebar.")

# Schedule next refresh if live
_maybe_rerun()
