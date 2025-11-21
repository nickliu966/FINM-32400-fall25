import os
import joblib
import argparse
import numpy as np
import pandas as pd
from typing import Literal


def load_models(model_path: str = "models") -> dict:
    models = {}
    for filename in os.listdir(model_path):
        if filename.endswith(".pkl"):
            exchange = filename.replace(".pkl", "")
            models[exchange] = joblib.load(f"models/{filename}")
    return models

MODELS = load_models()

def best_price_improvement(
    symbol: str,
    side: Literal["B", "S"],
    quantity: int,
    limit_price: float,
    bid_price: float,
    ask_price: float,
    bid_size: int,
    ask_size: int
) -> tuple[str, float]:
    """
    """

    # The ML model uses numeric side: Buy=1, Sell=0
    side_num = 1 if side == "B" else 0

    # create feature df to be input into models
    X = pd.DataFrame(
        [[side_num, quantity, limit_price,
        bid_price, ask_price, bid_size, ask_size]],
        columns=[
            "side_num", "order_qty", "limit_price",
            "bid_price", "ask_price", "bid_size", "ask_size"
        ]
    )

    best_exchange = None
    best_pi = -np.inf

    for exchange, model in MODELS.items():
        try:
            pred = model.predict(X)[0]
        except Exception:
            continue

        if pred > best_pi:
            best_pi = pred
            best_exchange = exchange

    return best_exchange, best_pi

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict best exchange for price improvement."
    )

    parser.add_argument("--symbol", type=str,
                        help="Symbol of the stock.")

    parser.add_argument("--side", type=str, choices=["B", "S"],
                        help="'B' for buy or 'S' for sell.")

    parser.add_argument("--quantity", type=int,
                        help="Order quantity.")

    parser.add_argument("--limit", type=float,
                        help="Limit price of the order.")

    parser.add_argument("--bid", type=float,
                        help="NBBO bid price.")

    parser.add_argument("--ask", type=float,
                        help="NBBO ask price.")

    parser.add_argument("--bid_size", type=int,
                        help="NBBO bid size.")

    parser.add_argument("--ask_size", type=int,
                        help="NBBO ask size.")

    return parser.parse_args()


def main():
    
    args = parse_args()

    exchange_id, pi = best_price_improvement(
        symbol=args.symbol,
        side=args.side,
        quantity=args.quantity,
        limit_price=args.limit,
        bid_price=args.bid,
        ask_price=args.ask,
        bid_size=args.bid_size,
        ask_size=args.ask_size
    )

    print(f"Best Exchange: {exchange_id}")
    print(f"Price Improvement: {pi}")

if __name__ == "__main__":
    main()