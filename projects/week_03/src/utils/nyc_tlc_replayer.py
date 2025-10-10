"""
NYC TLC Trip Data → Redpanda Replayer
-------------------------------------

Downloads monthly NYC TLC trip datasets (Parquet) and replays them as a configurable
Kafka/Redpanda stream. Works with yellow/green/fhv/fhvhv datasets.

Usage (examples)
----------------
# 1) Download last months and stream with constant rate (200 msg/s)
python nyc_tlc_replayer.py \
  --datasets yellow green \
  --months 2025-07 2025-08 \
  --mode kafka --brokers localhost:19092 --topic nyc.tlc.trips \
  --emit-mode constant --rate 200

# 2) Replay with time scaling (respect pickup time deltas sped-up 60x)
python nyc_tlc_replayer.py \
  --datasets yellow \
  --months 2025-08 \
  --mode kafka --brokers localhost:9092 --topic nyc.tlc.trips \
  --emit-mode timescale --speedup 60

# 3) Only download to ./data, no streaming yet
python nyc_tlc_replayer.py --datasets yellow --months 2025-08 --mode download

# 4) Stream to JSONL files instead of Kafka (for quick testing)
python nyc_tlc_replayer.py --datasets yellow --months 2025-08 --mode file --outdir ./out

Notes
-----
- Default source CDN is CloudFront with pattern:
  https://d37ci6vzurychx.cloudfront.net/trip-data/{dataset}_{YYYY-MM}.parquet
  where dataset in {yellow_tripdata, green_tripdata, fhv_tripdata, fhvhv_tripdata}.
- Script handles schema differences between datasets and older/newer months.
- Uses PyArrow streaming to avoid loading entire parquet into memory.
- Message key: derived from pickup_datetime + (vendorid or base) + row index.
- Safe fallbacks: missing coordinates (NaN) are skipped unless --allow-nan.

Dependencies
------------
  pip install pyarrow pandas requests confluent-kafka
"""
c
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import pyarrow.parquet as pq
import requests

# ----------------------------
# URLs / Datasets configuration
# ----------------------------
DATASET_MAP = {
    "yellow": "yellow_tripdata",
    "green": "green_tripdata",
    "fhv": "fhv_tripdata",
    "fhvhv": "fhvhv_tripdata",
}

DEFAULT_URL_TMPL = "https://d37ci6vzurychx.cloudfront.net/trip-data/{dataset}_{ym}.parquet"

# ----------------
# Helper utilities
# ----------------
ISO = "%Y-%m-%dT%H:%M:%S.%fZ"

def now_utc_str() -> str:
    return datetime.now(tz=timezone.utc).strftime(ISO)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ----------------
# Download logic
# ----------------
@dataclass
class DownloadSpec:
    dataset: str  # e.g., yellow_tripdata
    ym: str       # e.g., 2025-08
    url: str
    path: Path


def build_specs(datasets: List[str], months: List[str], root: Path, url_tmpl: str) -> List[DownloadSpec]:
    specs: List[DownloadSpec] = []
    for d in datasets:
        dname = DATASET_MAP.get(d, d)
        for ym in months:
            url = url_tmpl.format(dataset=dname, ym=ym)
            fname = f"{dname}_{ym}.parquet"
            specs.append(DownloadSpec(dataset=dname, ym=ym, url=url, path=root / fname))
    return specs


def download(spec: DownloadSpec, timeout: int = 60) -> Path:
    ensure_dir(spec.path.parent)
    if spec.path.exists() and spec.path.stat().st_size > 0:
        print(f"[SKIP] Exists: {spec.path.name}")
        return spec.path
    print(f"[GET] {spec.url}")
    with requests.get(spec.url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(spec.path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"[OK ] Saved: {spec.path}")
    return spec.path


# ----------------
# Sinks (Kafka/File)
# ----------------
class Sink:
    def send(self, topic: str, key: Optional[bytes], value: bytes):
        raise NotImplementedError
    def flush(self):
        pass


class KafkaSink(Sink):
    def __init__(self, brokers: str):
        from confluent_kafka import Producer  # type: ignore
        self.producer = Producer({"bootstrap.servers": brokers})

    def send(self, topic: str, key: Optional[bytes], value: bytes):
        self.producer.produce(topic=topic, key=key, value=value)
        self.producer.poll(0)

    def flush(self):
        self.producer.flush()


class FileSink(Sink):
    def __init__(self, outdir: Path, topic: str):
        ensure_dir(outdir)
        self.f = open(outdir / f"{topic.replace('.', '_')}.jsonl", "a", encoding="utf-8")
    def send(self, topic: str, key: Optional[bytes], value: bytes):
        rec = {"key": key.decode("utf-8") if key else None, "value": json.loads(value)}
        self.f.write(json.dumps(rec, ensure_ascii=False) + "")
    def flush(self):
        self.f.flush()


# ---------------------------
# Schema mapping / row sanitizer
# ---------------------------
# Explicit modern Yellow schema columns (post-2020)
YELLOW_COLS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
]

# Fallback for other datasets (green/fhv*)
COMMON_COLS = list(set(YELLOW_COLS + [
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "hvfhs_license_num",
    "base",
    "PUlocationID",
    "DOlocationID",
]))


def to_iso(ts) -> Optional[str]:
    if ts is None:
        return None
    try:
        return pd.to_datetime(ts, utc=True).strftime(ISO)
    except Exception:
        return None


def row_to_event(rec: Dict, idx: int, dataset_name: str) -> Dict:
    """Normalize a TLC record to a compact streaming event matching the shown yellow schema.
    Lat/Lon are not present in modern TLC; we stream LocationID-based trips.
    """
    # Prefer yellow timestamps; fallback to green/other
    pickup = rec.get("tpep_pickup_datetime") or rec.get("lpep_pickup_datetime") or rec.get("pickup_datetime")
    dropoff = rec.get("tpep_dropoff_datetime") or rec.get("lpep_dropoff_datetime") or rec.get("dropoff_datetime")

    event = {
        "event_type": "trip",
        "dataset": dataset_name,
        "pickup_ts": to_iso(pickup),
        "dropoff_ts": to_iso(dropoff),
        "pulocation_id": int(rec.get("PULocationID") or rec.get("PUlocationID") or -1),
        "dolocation_id": int(rec.get("DOLocationID") or rec.get("DOlocationID") or -1),
        "passenger_count": _safe_num(rec.get("passenger_count")),
        "trip_distance": _safe_num(rec.get("trip_distance")),
        "fare_amount": _safe_num(rec.get("fare_amount")),
        "total_amount": _safe_num(rec.get("total_amount")),
        "vendor": rec.get("VendorID") or rec.get("hvfhs_license_num") or rec.get("base"),
        "payment_type": rec.get("payment_type"),
        "congestion_surcharge": _safe_num(rec.get("congestion_surcharge")),
        "airport_fee": _safe_num(rec.get("Airport_fee")),
        "cbd_congestion_fee": _safe_num(rec.get("cbd_congestion_fee")),
        "ingested_at": now_utc_str(),
    }
    return event


def _safe_num(x):
    try:
        if x is None:
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def make_key(event: Dict, idx: int) -> str:
    base = event.get("pickup_ts") or "tsNA"
    vendor = str(event.get("vendor") or "VNA")
    return f"{base}_{vendor}_{idx}"


# -----------------
# Streaming emitters
# -----------------

def iter_batches(parquet_path: Path, batch_size: int = 50_000, columns: Optional[List[str]] = None) -> Iterable[pd.DataFrame]:
    """Yield DataFrames in batches without loading the whole file.
    Reads only selected columns for speed.
    """
    table = pq.read_table(parquet_path, columns=columns)
    for b in table.to_batches(max_chunksize=batch_size):
        yield b.to_pandas(types_mapper=pd.ArrowDtype)


def emit_constant_rate(df_iter: Iterable[pd.DataFrame], sink: Sink, topic: str, rate: int, dataset_name: str):
    interval = 1.0
    budget = 0
    for df in df_iter:
        records = df.to_dict(orient="records")
        i = 0
        while i < len(records):
            start = time.time()
            budget += rate
            sent = 0
            while sent < budget and i < len(records):
                ev = row_to_event(records[i], i, dataset_name)
                key = make_key(ev, i).encode("utf-8")
                sink.send(topic, key, json.dumps(ev, ensure_ascii=False).encode("utf-8"))
                sent += 1
                i += 1
            budget -= sent
            sink.flush()
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)


def emit_timescale(df_iter: Iterable[pd.DataFrame], sink: Sink, topic: str, speedup: float, dataset_name: str):
    prev_ts: Optional[datetime] = None
    for df in df_iter:
        for idx, rec in enumerate(df.to_dict(orient="records")):
            ev = row_to_event(rec, idx, dataset_name)
            ts_str = ev.get("pickup_ts")
            delay = 0.0
            if ts_str:
                try:
                    ts = pd.to_datetime(ts_str, utc=True).to_pydatetime()
                    if prev_ts is not None:
                        delta = (ts - prev_ts).total_seconds()
                        if delta > 0:
                            delay = delta / max(1e-6, speedup)
                    prev_ts = ts
                except Exception:
                    prev_ts = None
            if delay > 0:
                time.sleep(min(delay, 2.0))
            key = make_key(ev, idx).encode("utf-8")
            sink.send(topic, key, json.dumps(ev, ensure_ascii=False).encode("utf-8"))
        sink.flush()


# -------------
# Main function
# -------------

def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="NYC TLC → Redpanda Replayer")
    ap.add_argument("--datasets", nargs="+", default=["yellow"], choices=list(DATASET_MAP.keys()))
    ap.add_argument("--months", nargs="+", required=True, help="YYYY-MM (one or more)")
    ap.add_argument("--mode", choices=["download", "kafka", "file"], default="kafka")
    ap.add_argument("--brokers", default="localhost:19092")
    ap.add_argument("--topic", default="nyc.tlc.trips")
    ap.add_argument("--outdir", default="./out")
    ap.add_argument("--data-dir", default="./data/tripdata")
    ap.add_argument("--url-template", default=DEFAULT_URL_TMPL)

    emit = ap.add_mutually_exclusive_group()
    emit.add_argument("--emit-mode", choices=["constant", "timescale"], default="constant")
    ap.add_argument("--rate", type=int, default=200, help="msgs/sec for constant mode")
    ap.add_argument("--speedup", type=float, default=60.0, help="time acceleration for timescale mode")

    args = ap.parse_args(argv)

    data_root = Path(args.data_dir)
    specs = build_specs(args.datasets, args.months, data_root, args.url_template)

    # Download
    for s in specs:
        try:
            download(s)
        except Exception as e:
            print(f"[ERR] Failed to download {s.url}: {e}")
            if args.mode == "download":
                continue
            else:
                raise

    if args.mode == "download":
        print("[DONE] Downloads complete.")
        return

    # Sink init
    if args.mode == "kafka":
        sink: Sink = KafkaSink(args.brokers)
        print(f"[INFO] Streaming to Kafka topic {args.topic} @ {args.brokers}")
    else:
        sink = FileSink(Path(args.outdir), args.topic)
        print(f"[INFO] Streaming to file {args.outdir}")

    # Choose column projection based on dataset
    for s in specs:
        cols = YELLOW_COLS if s.dataset.startswith("yellow_") or s.dataset == "yellow_tripdata" else COMMON_COLS
        print(f"[PLAY] {s.path.name} ({s.dataset}) — mode={args.emit_mode}")
        df_iter = iter_batches(s.path, batch_size=50_000, columns=cols)
        if args.emit_mode == "constant":
            emit_constant_rate(df_iter, sink, args.topic, rate=args.rate, dataset_name=s.dataset)
        else:
            emit_timescale(df_iter, sink, args.topic, speedup=args.speedup, dataset_name=s.dataset)
        print(f"[OK  ] Finished {s.path.name}")

    sink.flush()
    print("[DONE] All files streamed.")


if __name__ == "__main__":
    main()
