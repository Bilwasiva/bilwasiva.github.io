"""Exact sample removal for a tiny deterministic ridge-regression model.

Educational illustration only. This is NOT a general unlearning method for
neural networks, a privacy certification, or a production deletion service.

Model: y_hat = w0 + w1*x
Loss:  sum_i (y_i - [1,x_i] @ w)**2 + ridge * ||w||**2.
The feature map and penalty are fixed; no dataset-fitted preprocessing is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Example:
    record_id: str
    x: float
    y: float


class ExactRidgeUnlearner:
    """Stores aggregate sufficient statistics and per-row contribution data.

    Tombstones and request IDs are kept in process memory solely to make the
    demonstration deterministic and idempotent. Production needs transactional,
    durable, access-controlled records with retention and privacy controls.
    """

    def __init__(self, ridge: float = 0.4):
        if ridge <= 0 or not isfinite(ridge):
            raise ValueError("ridge must be finite and strictly positive")
        self.ridge = ridge
        self.xx00 = 0.0
        self.xx01 = 0.0
        self.xx11 = 0.0
        self.xy0 = 0.0
        self.xy1 = 0.0
        self.rows: dict[str, Example] = {}
        self.tombstones: set[str] = set()
        self.requests: dict[str, str] = {}

    def _accumulate(self, row: Example, sign: int) -> None:
        self.xx00 += sign
        self.xx01 += sign * row.x
        self.xx11 += sign * row.x * row.x
        self.xy0 += sign * row.y
        self.xy1 += sign * row.x * row.y

    def add(self, row: Example) -> None:
        if not row.record_id or not all(map(isfinite, (row.x, row.y))):
            raise ValueError("valid record ID and finite numeric fields required")
        if row.record_id in self.rows or row.record_id in self.tombstones:
            raise ValueError("duplicate or previously deleted record ID")
        self.rows[row.record_id] = row
        self._accumulate(row, +1)

    def forget(self, record_id: str, request_id: str) -> bool:
        """Return True on first deletion, False on safe retry.

        RecordID + requestID binding prevents a request token being reused
        to target a different example. The operation is not a durable transaction.
        """
        if not request_id:
            raise ValueError("request_id is required")
        existing = self.requests.get(request_id)
        if existing is not None:
            if existing != record_id:
                raise ValueError("request_id cannot be reused for another record")
            return False
        row = self.rows.pop(record_id, None)
        if row is None:
            if record_id in self.tombstones:
                self.requests[request_id] = record_id
                return False
            raise KeyError(f"unknown record: {record_id}")
        self._accumulate(row, -1)
        self.tombstones.add(record_id)
        self.requests[request_id] = record_id
        return True

    def coefficients(self) -> tuple[float, float]:
        a = self.xx00 + self.ridge
        b = self.xx01
        d = self.xx11 + self.ridge
        determinant = a * d - b * b
        if determinant <= 0:
            raise ArithmeticError("nonpositive normal-equation determinant")
        return ((d * self.xy0 - b * self.xy1) / determinant,
                (a * self.xy1 - b * self.xy0) / determinant)

    def predict(self, x: float) -> float:
        w0, w1 = self.coefficients()
        return w0 + w1 * x


def train_from_scratch(rows: list[Example], ridge: float = 0.4) -> ExactRidgeUnlearner:
    model = ExactRidgeUnlearner(ridge)
    for row in rows:
        model.add(row)
    return model


def example_rows() -> list[Example]:
    rows = [Example(f"r{i}", float(i), 1.0 + 2.0 * i + delta)
            for i, delta in enumerate([.05, -.10, .08, -.06, .12, -.02])]
    rows.append(Example("forget-me", 2.5, 16.0))
    return rows


def demonstration() -> None:
    rows = example_rows()
    model = train_from_scratch(rows)
    before = model.coefficients()
    changed = model.forget("forget-me", "deletion-request-001")
    updated = model.coefficients()
    baseline = train_from_scratch([r for r in rows if r.record_id != "forget-me"]).coefficients()
    max_error = max(abs(a - b) for a, b in zip(updated, baseline))
    print(f"deletion applied: {changed}")
    print(f"before deletion: intercept={before[0]:.6f}; slope={before[1]:.6f}")
    print(f"after deletion:  intercept={updated[0]:.6f}; slope={updated[1]:.6f}")
    print(f"clean retrain:    intercept={baseline[0]:.6f}; slope={baseline[1]:.6f}")
    print(f"max coefficient discrepancy: {max_error:.3g}")
    assert max_error < 1e-11
    print("A correct sample-removal update matched clean retraining numerically.")


if __name__ == "__main__":
    demonstration()