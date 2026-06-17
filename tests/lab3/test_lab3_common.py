from __future__ import annotations

import time

from lab3.common import (
    monitored_block,
    parse_nvidia_smi_memory,
    peak_from_samples,
)


def test_parse_nvidia_smi_memory_handles_single_value() -> None:
    # `--query-gpu=memory.used --format=csv,noheader,nounits` prints MiB per GPU, one per line.
    assert parse_nvidia_smi_memory("24576") == 24.0  # 24576 MiB -> 24 GiB


def test_parse_nvidia_smi_memory_picks_first_multi_gpu_line() -> None:
    assert parse_nvidia_smi_memory("1024\n2048\n") == 1.0


def test_parse_nvidia_smi_memory_returns_none_on_garbage() -> None:
    assert parse_nvidia_smi_memory("not a number") is None
    assert parse_nvidia_smi_memory("") is None


def test_peak_from_samples_takes_max_ignoring_none() -> None:
    assert peak_from_samples([None, 1.0, 3.0, 2.0]) == 3.0


def test_peak_from_samples_all_none_returns_none() -> None:
    assert peak_from_samples([None, None]) is None
    assert peak_from_samples([]) is None


def test_monitored_block_records_timing_and_peak() -> None:
    timings: dict[str, float] = {}
    peaks: dict[str, float] = {}
    counter = {"i": 0}

    def rising_sampler() -> float:
        counter["i"] += 1
        return float(counter["i"])  # 1, 2, 3, ... GiB

    with monitored_block("train", timings, peaks, sampler=rising_sampler, interval=0.01):
        time.sleep(0.05)

    assert timings["train"] > 0.0
    assert peaks["train"] >= 1.0


def test_monitored_block_keeps_peak_absent_without_gpu() -> None:
    timings: dict[str, float] = {}
    peaks: dict[str, float] = {}

    with monitored_block("train", timings, peaks, sampler=lambda: None, interval=0.01):
        time.sleep(0.03)

    assert timings["train"] > 0.0
    assert "train" not in peaks  # all samples None -> no peak recorded


def test_monitored_block_works_without_peaks_dict() -> None:
    timings: dict[str, float] = {}
    with monitored_block("x", timings, None, sampler=lambda: 4.0, interval=0.01):
        time.sleep(0.02)
    assert timings["x"] > 0.0
