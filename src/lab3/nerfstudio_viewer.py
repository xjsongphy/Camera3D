from __future__ import annotations

import sys


def _ensure_png_default(argv: list[str]) -> list[str]:
    """Default the web viewer to lossless PNG unless the caller overrides it.

    nerfstudio's viewer ships with jpeg as the default image format (lossy, for
    interactivity). We prefer png for preview clarity. Both ``lab3-viewer ...``
    and ``lab3 --view-run`` (which launches lab3-viewer) pick this up. An
    explicit ``--viewer.image-format ...`` on the command line, or -h/--help,
    wins and is left untouched.
    """
    if any(a in ("-h", "--help") for a in argv):
        return argv
    if any(a == "--viewer.image-format" or a.startswith("--viewer.image-format=") for a in argv):
        return argv
    return argv + ["--viewer.image-format", "png"]


def main() -> None:
    from lab3.warning_filters import install_third_party_warning_filters

    install_third_party_warning_filters()

    sys.argv = _ensure_png_default(sys.argv)

    from nerfstudio.scripts.viewer.run_viewer import entrypoint

    entrypoint()


if __name__ == "__main__":
    main()
