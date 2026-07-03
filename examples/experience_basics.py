"""Experience basics: metric primitives and the ``Experience`` facade.

Bind the column roles once with ``Experience`` and then summarize the book any
way you like with ``.by`` and ``.views`` -- no need to re-pass the claim,
premium, and exposure columns each time. The free-function metric primitives
(``loss_ratio``, ``per_exposure``, ``pure_premium``, ...) are also shown.

    pip install actuarialpy
    python experience_basics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import actuarialpy as ap  # noqa: E402
from _sample_data import sample_member_months  # noqa: E402

EXPENSE = ["total_claims", "pharmacy_rebates", "non_ffs_expenses"]  # rebates are negative
REVENUE = ["premium"]
EXPOSURE = ["member_months"]


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main() -> None:
    df = sample_member_months()

    section("1. Metric primitives (free functions on scalars)")
    print(f"loss_ratio(8.5M, 10M)        : {ap.loss_ratio(8_500_000, 10_000_000):.3f}")
    print(f"pure_premium(330k, 6k)       : {ap.pure_premium(330_000, 6_000):.2f}")
    print(f"per_exposure(330k, 6k member-months) : {ap.per_exposure(330_000, 6_000):.2f}")
    print(f"permissible_loss_ratio(0.14) : {ap.permissible_loss_ratio(0.14):.3f}")

    # Declare the roles ONCE; reuse the object for every cut below.
    exp = ap.Experience(
        df,
        expense=EXPENSE,
        revenue=REVENUE,
        exposure=EXPOSURE,
        date="incurred_date",
        profile="health",
    )

    section("2. exp.by(): book total")
    print(exp.by().to_string(index=False))

    section("3. exp.by('line_of_business'): experience by LOB")
    print(exp.by("line_of_business").to_string(index=False))

    section("4. exp.views(): several named cuts in one call")
    views = exp.views({"by_lob": "line_of_business", "by_group": "group_id"})
    for name, frame in views.items():
        print(f"\n--- {name} ---")
        print(frame.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
