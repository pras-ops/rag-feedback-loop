"""
Gate C — the REAL verifier for the coding-agent showcase.

Loads coding problems from OpenAI's HumanEval dataset (released under the MIT License, copyright OpenAI)
and runs model-generated code against their actual unit tests in an isolated subprocess with a timeout.
Returns 1.0 (all asserts pass) or 0.0 — this is the objective, execution-based feedback signal `s_gt`
that replaces Gate A's synthetic keyword matcher.

NOTE: this executes generated code. A subprocess + timeout is the standard HumanEval
approach and is fine for a local demo, but production should sandbox in a container.
"""

import json
import subprocess
import sys
from typing import List, Optional


def load_humaneval(path: str = "data/HumanEval.jsonl", limit: Optional[int] = None) -> List[dict]:
    rows = [json.loads(line) for line in open(path)]
    return rows[:limit] if limit else rows


def run_tests(problem: dict, completion: str, timeout: float = 10.0) -> float:
    """
    Assemble  prompt + completion + test + check(entry_point)  and run it.

    `completion` is what the model produces: either the function body (HumanEval
    convention) or a full re-definition of the function — both work because the
    prompt is a valid (docstring-only) function and a later def overrides it.

    Returns 1.0 if the process exits 0 (all asserts passed), else 0.0.
    """
    program = (
        problem["prompt"] + completion + "\n"
        + problem["test"] + "\n"
        + f"check({problem['entry_point']})\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return 1.0 if proc.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0


if __name__ == "__main__":
    probs = load_humaneval(limit=8)
    print(f"Loaded {len(probs)} problems. Self-test (canonical must pass, broken must fail):\n")
    passed_canonical = 0
    failed_broken = 0
    for p in probs:
        ok = run_tests(p, p["canonical_solution"])
        broken = run_tests(p, "    return None\n")
        passed_canonical += int(ok == 1.0)
        failed_broken += int(broken == 0.0)
        print(f"  {p['task_id']:<14} entry={p['entry_point']:<22} canonical={ok}  broken={broken}")
    n = len(probs)
    print(f"\ncanonical passed: {passed_canonical}/{n}   broken correctly failed: {failed_broken}/{n}")
    print("VERIFIER OK" if (passed_canonical == n and failed_broken == n) else "VERIFIER PROBLEM")
