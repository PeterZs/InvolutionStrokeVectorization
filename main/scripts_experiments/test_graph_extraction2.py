"""Test runner for graph_extraction2 against the labelled cases under
debug_files/testcases. Each <name>.png is run through graph_extraction2.process
and the resulting graph's intersection / component counts are compared to
<name>.json: {"intersections": N, "components": M}.

Run:
    uv run test_graph_extraction2.py [testcase_dir]

Exit code is the number of failing tests (0 on full pass).
"""
import json
import sys
from pathlib import Path
import networkx as nx

import graph_extraction2


def run_tests(testcase_dir: Path) -> int:
    test_files = sorted(testcase_dir.glob("*.json"))
    if not test_files:
        print(f"No tests found in {testcase_dir}")
        return 0

    results = []
    for json_path in test_files:
        png_path = json_path.with_suffix(".png")
        if not png_path.exists():
            print(f"{json_path.stem}: SKIP (no matching .png)")
            continue

        with open(json_path) as f:
            expected = json.load(f)
        exp_int = expected.get("intersections")
        exp_comp = expected.get("components")

        print(f"\n=== {json_path.stem} ===")
        G = graph_extraction2.process(str(png_path))

        got_int = sum(1 for _, d in G.degree() if d >= 3)
        got_comp = nx.number_connected_components(G)

        ok_int = exp_int is None or got_int == exp_int
        ok_comp = exp_comp is None or got_comp == exp_comp
        passed = ok_int and ok_comp

        details = []
        if not ok_int:
            details.append(f"intersections {got_int} != {exp_int}")
        if not ok_comp:
            details.append(f"components {got_comp} != {exp_comp}")
        suffix = ("  --  " + "; ".join(details)) if details else (
            f"  (intersections={got_int}, components={got_comp})"
        )
        status = "PASS" if passed else "FAIL"
        results.append((json_path.stem, status, suffix))

    # Final summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    n_pass = 0
    n_fail = 0
    for name, status, suffix in results:
        print(f"  {name}: {status}{suffix}")
        if status == "PASS":
            n_pass += 1
        else:
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed (of {n_pass + n_fail})")
    return n_fail


if __name__ == "__main__":
    testcase_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "debug_files/testcases")
    failed = run_tests(testcase_dir)
    sys.exit(1 if failed else 0)
