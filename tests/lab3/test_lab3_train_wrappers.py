from __future__ import annotations

from lab3.gs_train_wrapper import ProgressFilter
from lab3.ns_train_wrapper import NerfstudioProgressFilter


def test_3dgs_progress_filter_keeps_console_progress_but_not_log_rewrites() -> None:
    filter_ = ProgressFilter()

    stdout_text, log_text = filter_.handle(
        "Training progress:  47%|███| 28140/60000 [00:01<00:01, Loss=0.0123456, Depth Loss=0.0012345]\n"
    )

    assert "Training progress: 28140/60000" in (stdout_text or "")
    assert log_text is None
    assert filter_.curves == [
        {
            "iteration": 28140,
            "total_iterations": 60000,
            "loss": 0.0123456,
            "depth_loss": 0.0012345,
        }
    ]


def test_nerfstudio_progress_filter_collapses_table_row_for_console() -> None:
    filter_ = NerfstudioProgressFilter()

    assert filter_.handle("Step (% Done)       Train Iter (time)    ETA (time)\n") is None
    assert filter_.handle("---------------------------------------------------\n") is None
    console = filter_.handle(
        "28140 (46.90%)      19.582 ms            10 m, 23 s           213.22 K             1.01 M\n"
    )

    assert console == (
        "\rNeRF train: step 28140 (46.90%) | iter 19.582 ms | ETA 10 m, 23 s | "
        "train 213.22 K | test 1.01 M"
    )


def test_nerfstudio_progress_filter_flushes_before_regular_line() -> None:
    filter_ = NerfstudioProgressFilter()
    filter_.handle("28140 (46.90%)      19.582 ms            10 m, 23 s           213.22 K\n")

    regular = filter_.handle("Saving config to disk\n")

    assert regular == "\nSaving config to disk\n"
