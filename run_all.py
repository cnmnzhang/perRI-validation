"""Dispatcher: run one or all analyses.

Run in a preferred order, since some analyses depend on others (e.g., fig3_dx and fig4_dx_cases depend on dx_incident).

Run:
    python -m run_all --analysis all
    python -m run_all --analysis dx_incident
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from utils.bootstrap import ensure_importable
from scripts import build_dx_incident, run_fig3_dx, run_fig3_hazard, run_fig4_dx_cases, run_fig4_pregnancy, run_fig5_iron_infusion, build_splits_by_marker

ensure_importable()

from utils.logging_utils import tagged_stdout  # noqa: E402
from scripts import build_setpoints  # noqa: E402

ANALYSES = {
    "tests_by_marker": build_splits_by_marker.run,
    "setpoints_by_marker": build_setpoints.run,
    "dx_incident": build_dx_incident.run,
    "fig3_hazard": run_fig3_hazard.run,
    "fig3_dx": run_fig3_dx.run,
    "fig4_dx_cases": run_fig4_dx_cases.run,
    "fig4_pregnancy": run_fig4_pregnancy.run,
    "fig5_iron_infusion": run_fig5_iron_infusion.run,
}

# Analyses that consume dx_incident's dx_incident.csv rather than re-deriving it.
_NEEDS_dx_incident = {"fig3_dx", "fig4_dx_cases"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run perri-validation analyses end to end.")
    parser.add_argument("--analysis", choices=[*ANALYSES, "all"], default="all")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "outputs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    targets = ANALYSES if args.analysis == "all" else {args.analysis: ANALYSES[args.analysis]}
    manifests = {}
    elapsed = {}
    for name, fn in targets.items():
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        kwargs = {"input_dir": args.input_dir, "output_dir": args.output_dir / name, "force": args.force}
        if name in _NEEDS_dx_incident:
            kwargs["dx_incident_path"] = args.output_dir / "dx_incident" / "dx_incident.csv"
        t0 = time.time()
        # Each analysis is independent -- one failing (bad input, a downstream
        # dependency like dx_incident missing, ...) shouldn't prevent the others
        # from running, especially once --analysis all has already spent time on
        # earlier analyses.
        try:
            with tagged_stdout(name):
                manifests[name] = fn(**kwargs)
        except Exception as exc:
            print(f"[{name}] SKIPPED, analysis failed: {exc}")
            manifests[name] = {"error": str(exc)}
        elapsed[name] = round(time.time() - t0, 1)
        print(f"[{name}] finished in {elapsed[name]}s")

    print(f"\n{'=' * 20} summary {'=' * 20}")
    print(f"elapsed seconds: {json.dumps(elapsed, indent=2, sort_keys=True)}")
    # print(json.dumps(manifests, indent=2, sort_keys=True, default=str))  # commented out -- clogs output; see save_fig_as_svg for per-figure Figure/Data lines instead


if __name__ == "__main__":
    main()
