#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from  confluent_kafka import Consumer, Producer, KafkaException, KafkaError

# ---- ВАШІ АЛГОРИТМИ ----
# покладіть probabilistic.py у PYTHONPATH або імпортуйте модулем з вашої структури
from src.data_structures.probabilistic import (
    BloomFilter,
    HyperLogLog,
    CountMinSketch,
    ReservoirSampling,
    MisraGries,
    MinHashLSH,
)

# -------------------- Helpers --------------------
def parse_event(value: Optional[bytes]) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}

def pickup_dt(ev: Dict[str, Any]) -> datetime:
    ts = ev.get("pickup_ts") or ev.get("ingested_at")
    if ts is None:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

def floor_to_minutes(dt: datetime, step_min: int) -> datetime:
    minute = (dt.minute // step_min) * step_min
    return dt.replace(minute=minute, second=0, microsecond=0)

def grid_hash(lat: Optional[float], lon: Optional[float], cell_deg: float = 0.01) -> Optional[str]:
    if lat is None or lon is None:
        return None
    gy = int(lat / cell_deg)
    gx = int(lon / cell_deg)
    return f"g{cell_deg:.3f}_{gy}_{gx}"

# -------------------- Windowed State --------------------
@dataclass
class AlgoState:
    bloom: BloomFilter
    hll_vendors: HyperLogLog
    hll_od: HyperLogLog
    cms: CountMinSketch
    mg: MisraGries
    reservoir: ReservoirSampling[Dict[str, Any]]
    lsh: MinHashLSH
    events: int = 0

    @staticmethod
    def new(capacity_hint: int, bloom_err: float, res_k: int) -> "AlgoState":
        return AlgoState(
            bloom=BloomFilter.from_capacity(capacity_hint, error_rate=bloom_err),
            hll_vendors=HyperLogLog(precision=14),
            hll_od=HyperLogLog(precision=14),
            cms=CountMinSketch(width=4096, depth=5),
            mg=MisraGries(k=50),
            reservoir=ReservoirSampling(k=res_k),
            lsh=MinHashLSH(num_hashes=128, threshold=0.5),
            events=0,
        )

class WindowManager:
    def __init__(
        self,
        windows_min: List[int],
        lateness_sec: int,
        capacity_hint: int,
        bloom_err: float,
        res_k: int,
        dedup: bool,
    ):
        self.windows_min = sorted(windows_min)
        self.lateness = timedelta(seconds=lateness_sec)
        self.capacity_hint = capacity_hint
        self.bloom_err = bloom_err
        self.res_k = res_k
        self.dedup = dedup
        # {win: {window_start: AlgoState}}
        self.state: Dict[int, Dict[datetime, AlgoState]] = {w: {} for w in self.windows_min}

    def _start_dt(self, dt: datetime, win: int) -> datetime:
        return floor_to_minutes(dt, win).astimezone(timezone.utc)

    def update(self, ev: Dict[str, Any]) -> None:
        dt = pickup_dt(ev)
        vendor = str(ev.get("vendor", "NA"))
        gh1 = grid_hash(ev.get("pickup_lat"), ev.get("pickup_lon"))
        gh2 = grid_hash(ev.get("dropoff_lat"), ev.get("dropoff_lon"))
        od_key = f"{gh1}->{gh2}"
        puloc = str(ev.get("pulocation_id") or "NA")

        bloom_key = f"{ev.get('pickup_ts','NA')}:{vendor}"

        # набір токенів для LSH (простий OD-відбиток)
        item_set = set(filter(None, [f"PU:{gh1}", f"DO:{gh2}"]))

        for win in self.windows_min:
            start = self._start_dt(dt, win)
            bucket = self.state[win].get(start)
            if bucket is None:
                bucket = AlgoState.new(self.capacity_hint, self.bloom_err, self.res_k)
                self.state[win][start] = bucket

            # Bloom
            seen = (bloom_key in bucket.bloom)
            bucket.bloom.add(bloom_key)
            if self.dedup and seen:
                continue

            # HLL / CMS / MG / Reservoir
            bucket.hll_vendors.add(vendor)
            bucket.hll_od.add(od_key)
            bucket.cms.add(od_key)
            bucket.mg.process_stream(iter([puloc]))      # MG — інкрементально через 1-елементний ітератор
            bucket.reservoir.process_stream(iter([{      # Reservoir — те саме
                "pickup_ts": ev.get("pickup_ts"),
                "pulocation_id": ev.get("pulocation_id"),
                "dolocation_id": ev.get("dolocation_id"),
                "vendor": ev.get("vendor"),
            }]))
            # LSH: порахуємо розмір кластера кандидатів (до додавання поточного)
            cluster_before = len(bucket.lsh.query(item_set))
            bucket.lsh.add(od_key, item_set)
            bucket.events += 1

    def flush_due(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        if now is None:
            now = datetime.now(timezone.utc)

        out: List[Dict[str, Any]] = []
        for win, buckets in list(self.state.items()):
            win_delta = timedelta(minutes=win)
            to_del: List[datetime] = []
            for start_dt, st in buckets.items():
                end_dt = start_dt + win_delta
                if end_dt + self.lateness < now:
                    out.append({
                        "window_min": win,
                        "start": start_dt.isoformat(),
                        "end": end_dt.isoformat(),
                        "events": st.events,
                        "hll": {
                            "vendors": int(st.hll_vendors.estimate()),
                            "od_pairs": int(st.hll_od.estimate()),
                        },
                        "mg_top5": sorted(st.mg.get_frequent_items().items(), key=lambda x: -x[1])[:5],
                        "reservoir_size": len(st.reservoir.get_sample()),
                        # невеликий сніпет для CMS: оцінка для останніх частих OD можна добирати на дашборді
                    })
                    to_del.append(start_dt)
            for k in to_del:
                del buckets[k]
        return out

# -------------------- Kafka runner --------------------
@dataclass
class Settings:
    brokers: List[str]
    topic_in: str
    group: str
    emit_topic: Optional[str]
    start_from_end: bool
    poll_timeout_s: float
    commit_every: int
    flush_interval_s: float
    lateness_sec: int
    windows_min: List[int]
    capacity_hint: int
    bloom_err: float
    reservoir_k: int
    dedup: bool

def _split(val: str) -> List[str]:
    return [p.strip() for p in val.split(",") if p.strip()]

def _mk_consumer(cfg: Settings) -> Consumer:
    return Consumer({
        "bootstrap.servers": ",".join(cfg.brokers),
        "group.id": cfg.group,
        "enable.auto.commit": False,
        "auto.offset.reset": "latest" if cfg.start_from_end else "earliest",
        "allow.auto.create.topics": True,
    })

def _mk_producer(cfg: Settings) -> Optional[Producer]:
    if not cfg.emit_topic:
        return None
    return Producer({"bootstrap.servers": ",".join(cfg.brokers)})

def _delivery_report(err, msg):
    if err is not None:
        sys.stderr.write(f"[PRODUCE_ERROR] {err}\n")

def run(cfg: Settings):
    stop = False
    def _sig(*_):
        nonlocal stop
        stop = True
        print("Signal received, stopping…")
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    c = _mk_consumer(cfg)
    p = _mk_producer(cfg)
    c.subscribe([cfg.topic_in])

    wm = WindowManager(
        windows_min=cfg.windows_min,
        lateness_sec=cfg.lateness_sec,
        capacity_hint=cfg.capacity_hint,
        bloom_err=cfg.bloom_err,
        res_k=cfg.reservoir_k,
        dedup=cfg.dedup,
    )

    last_flush = time.time()
    n_since_commit = 0
    try:
        while not stop:
            msg = c.poll(cfg.poll_timeout_s)
            if msg is None:
                pass
            elif msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    raise KafkaException(msg.error())
            else:
                ev = parse_event(msg.value())
                wm.update(ev)
                n_since_commit += 1
                if n_since_commit >= cfg.commit_every:
                    c.commit(asynchronous=True)
                    n_since_commit = 0

            now = time.time()
            if now - last_flush >= cfg.flush_interval_s:
                outputs = wm.flush_due()
                for out in outputs:
                    line = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        **out,
                    }
                    key = f"{line['window_min']}:{line['start']}".encode("utf-8")
                    val = json.dumps(line).encode("utf-8")
                    # console
                    print(f"[WIN {line['window_min']}m] {line['start']}..{line['end']} "
                          f"events={line['events']} vendors≈{line['hll']['vendors']} "
                          f"od_pairs≈{line['hll']['od_pairs']} top5={line['mg_top5']} "
                          f"reservoir={line['reservoir_size']}")
                    # emit
                    if p and cfg.emit_topic:
                        p.produce(cfg.emit_topic, key=key, value=val, on_delivery=_delivery_report)
                if p:
                    p.poll(0)
                last_flush = now
    finally:
        try:
            c.commit(asynchronous=False)
        except Exception:
            pass
        c.close()
        if p:
            p.flush(5.0)

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brokers", default="localhost:19092", help="comma-separated")
    ap.add_argument("--topic-in", default="nyc.tlc.trips")
    ap.add_argument("--group", default="tlc-metrics")
    ap.add_argument("--emit-topic", default="tlc.metrics", help="empty to disable")
    ap.add_argument("--start-from-end", default="true", choices=["true", "false"])
    ap.add_argument("--poll-timeout", type=float, default=0.5)
    ap.add_argument("--commit-every", type=int, default=500)
    ap.add_argument("--flush-interval", type=float, default=5.0)
    ap.add_argument("--lateness-sec", type=int, default=30)
    ap.add_argument("--windows", default="1,5,10")
    ap.add_argument("--capacity-hint", type=int, default=500_000, help="для Bloom.from_capacity")
    ap.add_argument("--bloom-err", type=float, default=0.005)
    ap.add_argument("--reservoir-k", type=int, default=50)
    ap.add_argument("--dedup", default="false", choices=["true", "false"])
    args = ap.parse_args()

    cfg = Settings(
        brokers=_split(args.brokers),
        topic_in=args.topic_in,
        group=args.group,
        emit_topic=(args.emit_topic if args.emit_topic.strip() else None),
        start_from_end=(args.start_from_end == "true"),
        poll_timeout_s=max(0.05, args.poll_timeout),
        commit_every=max(1, args.commit_every),
        flush_interval_s=max(0.2, args.flush_interval),
        lateness_sec=max(0, args.lateness_sec),
        windows_min=[int(x) for x in args.windows.split(",") if x.strip()],
        capacity_hint=args.capacity_hint,
        bloom_err=args.bloom_err,
        reservoir_k=args.reservoir_k,
        dedup=(args.dedup == "true"),
    )
    run(cfg)

if __name__ == "__main__":
    main()
