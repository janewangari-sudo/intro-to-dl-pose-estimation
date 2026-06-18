"""Compare persisted Average-Pose and SimpleBaseline PCK results.

Run from the repository root after evaluating both methods:
    python scripts/compare_baselines.py \
        --config configs/coco_simplebaseline.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, output_path, project_path


TRIVIAL_RESULTS_PATH = "outputs/results/trivial_baseline_pck.json"
COMPARISON_PATH = "outputs/results/baseline_comparison.json"


class ComparisonError(Exception):
    """Raised when persisted evaluation results cannot be compared."""


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/coco_simplebaseline.yaml",
        help="YAML configuration path relative to the repository root.",
    )
    parser.add_argument(
        "--trivial-results",
        default=TRIVIAL_RESULTS_PATH,
        help="Average-Pose PCK JSON path relative to the repository root.",
    )
    parser.add_argument(
        "--simple-results",
        default=None,
        help=(
            "SimpleBaseline PCK JSON path. Defaults to output.pck_results "
            "from the configuration."
        ),
    )
    parser.add_argument(
        "--output",
        default=COMPARISON_PATH,
        help="Comparison JSON path under outputs/.",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    """Return a repository-relative path when possible."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved_path)


def load_result(path: Path, label: str) -> dict[str, Any]:
    """Load and minimally validate one evaluator's JSON output."""
    try:
        with path.open("r", encoding="utf-8") as result_file:
            result = json.load(result_file)
    except json.JSONDecodeError as error:
        raise ComparisonError(
            f"{label} is not valid JSON: {display_path(path)} ({error})"
        ) from error

    if not isinstance(result, dict):
        raise ComparisonError(f"{label} must contain a JSON object.")

    required = ("mean_pck", "threshold", "num_valid_keypoints")
    missing = [field for field in required if field not in result]
    if missing:
        raise ComparisonError(
            f"{label} is missing {', '.join(missing)}. Re-run its evaluation "
            "script to regenerate the result JSON."
        )

    for field in ("mean_pck", "threshold"):
        value = result[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ComparisonError(f"{label} has an invalid '{field}' value.")
        result[field] = float(value)

    valid_count = result["num_valid_keypoints"]
    if (
        isinstance(valid_count, bool)
        or not isinstance(valid_count, int)
        or valid_count < 0
    ):
        raise ComparisonError(
            f"{label} has an invalid 'num_valid_keypoints' value."
        )

    per_joint = result.get("per_joint_pck")
    if per_joint is not None and not isinstance(per_joint, dict):
        raise ComparisonError(
            f"{label} field 'per_joint_pck' must be a JSON object."
        )
    per_joint_counts = result.get("per_joint_valid_keypoints")
    if per_joint_counts is not None and not isinstance(
        per_joint_counts,
        dict,
    ):
        raise ComparisonError(
            f"{label} field 'per_joint_valid_keypoints' must be a JSON "
            "object."
        )
    return result


def compare_results(
    trivial: dict[str, Any],
    simple: dict[str, Any],
    trivial_path: Path,
    simple_path: Path,
) -> dict[str, Any]:
    """Verify a fair comparison and build its JSON payload."""
    if not math.isclose(
        trivial["threshold"],
        simple["threshold"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ComparisonError(
            "PCK thresholds differ: Average-Pose uses "
            f"{trivial['threshold']}, while SimpleBaseline uses "
            f"{simple['threshold']}."
        )

    trivial_count = trivial["num_valid_keypoints"]
    simple_count = simple["num_valid_keypoints"]
    if trivial_count != simple_count:
        raise ComparisonError(
            "Valid-keypoint counts differ: Average-Pose evaluated "
            f"{trivial_count}, while SimpleBaseline evaluated {simple_count}."
        )

    trivial_joint_counts = trivial.get("per_joint_valid_keypoints")
    simple_joint_counts = simple.get("per_joint_valid_keypoints")
    if (
        trivial_joint_counts is not None
        and simple_joint_counts is not None
        and trivial_joint_counts != simple_joint_counts
    ):
        raise ComparisonError(
            "Per-joint valid-keypoint counts differ. Re-run both evaluations "
            "with the same dataset configuration."
        )

    trivial_scores = trivial.get("per_joint_pck")
    simple_scores = simple.get("per_joint_pck")
    per_joint_difference = None
    if trivial_scores is not None and simple_scores is not None:
        if trivial_scores.keys() != simple_scores.keys():
            raise ComparisonError(
                "Per-joint PCK names differ between the result files."
            )
        try:
            per_joint_difference = {
                name: (
                    None
                    if trivial_score is None or simple_scores[name] is None
                    else float(simple_scores[name]) - float(trivial_score)
                )
                for name, trivial_score in trivial_scores.items()
            }
        except (TypeError, ValueError) as error:
            raise ComparisonError(
                "Per-joint PCK values must be numeric or null."
            ) from error

    return {
        "comparison": "trivial_average_pose_vs_simplebaseline",
        "pck_threshold": trivial["threshold"],
        "num_valid_keypoints": trivial_count,
        "per_joint_valid_keypoints": (
            trivial_joint_counts
            if trivial_joint_counts == simple_joint_counts
            else None
        ),
        "methods": {
            "trivial_average_pose": {
                "mean_pck": trivial["mean_pck"],
                "per_joint_pck": trivial_scores,
                "source_results": display_path(trivial_path),
            },
            "simplebaseline": {
                "mean_pck": simple["mean_pck"],
                "per_joint_pck": simple_scores,
                "source_results": display_path(simple_path),
            },
        },
        "simplebaseline_minus_trivial": {
            "mean_pck": simple["mean_pck"] - trivial["mean_pck"],
            "per_joint_pck": per_joint_difference,
        },
    }


def run(args: argparse.Namespace) -> Path:
    """Load both evaluations, compare them, and save the result JSON."""
    config = load_config(args.config)
    output_config = config["output"]
    output_dir = output_path(output_config["dir"])
    trivial_path = project_path(args.trivial_results)
    simple_path = (
        project_path(args.simple_results)
        if args.simple_results
        else output_dir / output_config["pck_results"]
    )
    weights_path = output_dir / output_config["weights"]

    if not trivial_path.is_file():
        raise ComparisonError(
            "Average-Pose PCK result not found: "
            f"{display_path(trivial_path)}.\n"
            "Run:\n"
            f"  python scripts/evaluate_trivial_baseline.py --config "
            f"{args.config}"
        )
    if not simple_path.is_file():
        checkpoint_status = (
            f"A checkpoint exists at {display_path(weights_path)}."
            if weights_path.is_file()
            else (
                "No checkpoint was found at the configured path "
                f"{display_path(weights_path)}."
            )
        )
        raise ComparisonError(
            "SimpleBaseline PCK result not found: "
            f"{display_path(simple_path)}. {checkpoint_status}\n"
            "Run:\n"
            f"  python scripts/evaluate_pck.py --config {args.config}\n"
            "Use --weights PATH with that command if the trained checkpoint "
            "is stored elsewhere."
        )

    comparison = compare_results(
        load_result(trivial_path, "Average-Pose result"),
        load_result(simple_path, "SimpleBaseline result"),
        trivial_path,
        simple_path,
    )
    comparison_path = output_path(args.output)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", encoding="utf-8") as output_file:
        json.dump(comparison, output_file, indent=2, allow_nan=False)

    methods = comparison["methods"]
    difference = comparison["simplebaseline_minus_trivial"]["mean_pck"]
    print(
        "Average-Pose mean PCK: "
        f"{methods['trivial_average_pose']['mean_pck'] * 100:.1f}%"
    )
    print(
        "SimpleBaseline mean PCK: "
        f"{methods['simplebaseline']['mean_pck'] * 100:.1f}%"
    )
    print(f"Difference: {difference * 100:+.1f} percentage points")
    print(
        f"PCK threshold: {comparison['pck_threshold']} | "
        f"Valid keypoints: {comparison['num_valid_keypoints']}"
    )
    print(f"Comparison saved to {comparison_path}")
    return comparison_path


def main() -> int:
    """Run the workflow without exposing expected errors as tracebacks."""
    try:
        run(parse_args())
    except (ComparisonError, OSError, KeyError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
