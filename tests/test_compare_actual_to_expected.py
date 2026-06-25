"""Tests for two-table actual-vs-expected comparison, incl. column-name collisions."""

import numpy as np
import pandas as pd

from actuarialpy.forecast import compare_actual_to_expected


def _two_tables():
    """Actuals and forecast as two separate frames, both using an 'amount' column,
    multi-column keys, and actuals missing for future months."""
    months = pd.date_range("2024-01-01", "2024-06-01", freq="MS")
    keys = ["month", "segment", "category"]

    def frame(amount, through=None):
        rows = []
        for m in months:
            if through is not None and m > through:
                continue
            for seg in ("EHPI", "HIP"):
                for cat in ("Expenses", "Member Months"):
                    rows.append(dict(month=m, segment=seg, category=cat, amount=amount))
        return pd.DataFrame(rows)

    actuals = frame(110.0, through=pd.Timestamp("2024-04-01"))
    forecast = frame(100.0)
    return actuals, forecast, keys


def test_amount_collision_produces_suffixed_columns():
    actuals, forecast, keys = _two_tables()
    out = compare_actual_to_expected(actuals, forecast, on=keys,
                                     actual_col="amount", expected_col="amount", how="outer")
    assert {"amount_actual", "amount_expected"}.issubset(out.columns)
    assert "amount_x" not in out.columns and "amount_y" not in out.columns


def test_collision_variance_math():
    actuals, forecast, keys = _two_tables()
    out = compare_actual_to_expected(actuals, forecast, on=keys,
                                     actual_col="amount", expected_col="amount", how="outer")
    row = out[(out.month == "2024-04-01") & (out.segment == "EHPI")
              & (out.category == "Expenses")].iloc[0]
    assert row.amount_actual == 110.0
    assert row.amount_expected == 100.0
    assert row.variance == 10.0
    assert np.isclose(row.actual_to_expected, 1.10)
    assert np.isclose(row.variance_pct, 0.10)


def test_outer_join_missing_actual_is_nan():
    actuals, forecast, keys = _two_tables()
    out = compare_actual_to_expected(actuals, forecast, on=keys,
                                     actual_col="amount", expected_col="amount", how="outer")
    fut = out[(out.month == "2024-05-01") & (out.segment == "EHPI")
              & (out.category == "Expenses")].iloc[0]
    assert pd.isna(fut.amount_actual)
    assert pd.isna(fut.variance)
    assert fut.amount_expected == 100.0


def test_outer_join_keeps_all_keys():
    actuals, forecast, keys = _two_tables()
    out = compare_actual_to_expected(actuals, forecast, on=keys,
                                     actual_col="amount", expected_col="amount", how="outer")
    assert len(out) == len(forecast)  # 24 forecast keys all retained


def test_custom_suffixes():
    actuals, forecast, keys = _two_tables()
    out = compare_actual_to_expected(actuals, forecast, on=keys,
                                     actual_col="amount", expected_col="amount",
                                     how="outer", suffixes=("actual", "forecast"))
    assert {"amount_actual", "amount_forecast"}.issubset(out.columns)
    assert "amount_expected" not in out.columns


def test_distinct_names_unchanged():
    # backward compatibility: when names differ, nothing is renamed
    a = pd.DataFrame({"id": [1, 2, 3], "actual": [6000, 100, 50]})
    e = pd.DataFrame({"id": [1, 2, 3], "expected": [5500, 120, 40]})
    out = compare_actual_to_expected(a, e, on="id", actual_col="actual", expected_col="expected")
    assert "actual" in out.columns and "expected" in out.columns
    assert "actual_actual" not in out.columns
    assert np.isclose(out.loc[0, "actual_to_expected"], 6000 / 5500)


def test_case_b_unrelated_collision_renames_expected_only():
    # actual has a stray column that happens to match the expected amount-col name
    a = pd.DataFrame({"id": [1, 2], "amount": [10, 20], "forecast": [1, 2]})
    e = pd.DataFrame({"id": [1, 2], "forecast": [8, 18]})
    out = compare_actual_to_expected(a, e, on="id", actual_col="amount", expected_col="forecast")
    assert "forecast_expected" in out.columns  # expected amount disambiguated
    assert "amount" in out.columns              # actual amount left as-is
    assert out.loc[0, "variance"] == 10 - 8


def test_multi_column_keys():
    actuals, forecast, keys = _two_tables()
    assert len(keys) == 3
    out = compare_actual_to_expected(actuals, forecast, on=keys,
                                     actual_col="amount", expected_col="amount", how="outer")
    # one row per (month, segment, category)
    assert len(out) == out[keys].drop_duplicates().shape[0]


def test_default_left_join_backward_compatible():
    a = pd.DataFrame({"id": [1, 2], "actual": [10, 20]})
    e = pd.DataFrame({"id": [1, 2, 3], "expected": [8, 18, 28]})
    out = compare_actual_to_expected(a, e, on="id", actual_col="actual", expected_col="expected")
    assert len(out) == 2  # left join keeps only actual's keys


def test_metric_order_is_variance_then_ratio_last():
    a = pd.DataFrame({"id": [1, 2], "amount": [110.0, 120.0]})
    e = pd.DataFrame({"id": [1, 2], "amount": [100.0, 100.0]})
    out = compare_actual_to_expected(a, e, on="id", actual_col="amount", expected_col="amount")
    assert list(out.columns)[-3:] == ["variance", "variance_pct", "actual_to_expected"]
