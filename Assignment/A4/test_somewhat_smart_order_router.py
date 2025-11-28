import numpy as np
import pytest

import somewhat_smart_order_router as ssor


# create a dummy model with predictable output
class DummyModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        # returns a constant dummy price improvement
        return np.array([self.value])


def test_select_highest_improvement():
    """
    Test that the router picks the exchange with the highest predicted PI.
    """

    # Mock 3 exchanges with predictable outputs
    ssor.MODELS = {
        "ID1": DummyModel(0.1),
        "ID2": DummyModel(0.5),
        "ID3": DummyModel(0.3),
    }

    exchange, pi = ssor.best_price_improvement(
        symbol="AAPL",
        side="B",
        quantity=100,
        limit_price=250.0,
        bid_price=249.5,
        ask_price=250.2,
        bid_size=1000,
        ask_size=800
    )

    assert exchange == "ID2"
    assert pi == pytest.approx(0.5)


def test_invalid_input_side():
    """
    Passing an invalid side (not 'B' or 'S') should raise ValueError.
    """
    with pytest.raises(ValueError):
        ssor.best_price_improvement(
            symbol="AAPL",
            side="X",          # invalid side
            quantity=100,
            limit_price=250.0,
            bid_price=249.5,
            ask_price=250.2,
            bid_size=500,
            ask_size=600
        )
