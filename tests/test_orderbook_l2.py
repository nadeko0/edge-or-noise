"""Order-book reader checks: every snap50 line is the whole book, so a
level missing from a later line must be gone, not carried over from
an earlier one (the bug that made the original L2 features unusable)."""
import gzip
import json

from tickml.orderbook_l2 import _load_snap50_day


def _write(path, lines):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for o in lines:
            fh.write(json.dumps(o) + "\n")


def test_level_absent_from_later_snapshot_is_dropped(tmp_path):
    p = tmp_path / "BTCUSDT_2026-01-01.jsonl.gz"
    _write(p, [
        # minute 0: bid 101 exists; price then drops and 101 leaves the book
        {"T": 0, "b": [["101", "1"], ["100", "1"]], "a": [["102", "1"]]},
        {"T": 1000, "b": [["99", "2"], ["98", "2"]], "a": [["100", "3"]]},
        # minute 1 starts -> minute 0 row is emitted from the last line above
        {"T": 60_000, "b": [["99", "2"]], "a": [["100", "3"]]},
    ])
    rows = _load_snap50_day(p)
    assert len(rows) == 2
    first = rows[0]
    assert first["n_bid_levels"] == 2 and first["n_ask_levels"] == 1
    assert first["top5_bid"] == 4.0  # 99 + 98, not including the stale 101/100
    assert first["top5_ask"] == 3.0
