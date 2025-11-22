import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

import argparse
import joblib
from tqdm import tqdm


def read_executions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Convert timestamps: 20250910-04:02:34.713
    df['OrderTransactTime'] = pd.to_datetime(
        df['OrderTransactTime'], 
        format='%Y%m%d-%H:%M:%S.%f'
    )
    df['ExecutionTransactTime'] = pd.to_datetime(
        df['ExecutionTransactTime'], 
        format='%Y%m%d-%H:%M:%S.%f'
    )

    # Consistent column names
    df = df.rename(columns={
        "OrderTransactTime": "order_time",
        "ExecutionTransactTime": "execution_time",
        "AvgPx": "execution_price",
        "LastMkt": "exchange",
        "Symbol": "symbol",
        "OrderQty": "order_qty",
        "LimitPrice": "limit_price"
    })

    # Make symbol categorical
    df['symbol'] = df['symbol'].astype('category')

    # Side as string ('1' = buy)
    df['Side'] = df['Side'].astype(str)

    # Filter to market hours: 09:30–16:00
    market_open = pd.to_datetime("09:30").time()
    market_close = pd.to_datetime("16:00").time()

    df = df[
        (df['order_time'].dt.time >= market_open) &
        (df['order_time'].dt.time <= market_close)
    ]

    return df


def read_quotes(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, compression="gzip")

    # sip_timestamp convert to datetime
    df['sip_timestamp'] = pd.to_datetime(df['sip_timestamp'], unit='ns')

    # Convert ticker to category
    df['ticker'] = df['ticker'].astype('category')

    # Filter market hours
    market_open = pd.to_datetime("09:30").time()
    market_close = pd.to_datetime("16:00").time()

    df = df[
        (df['sip_timestamp'].dt.time >= market_open) &
        (df['sip_timestamp'].dt.time <= market_close)
    ]

    print("read quotes successfule")
    return df


def merge_execution_and_quotes(exec_df: pd.DataFrame, quotes_df: pd.DataFrame) -> pd.DataFrame:
    
    exec_df["symbol"] = exec_df["symbol"].astype(str)
    quotes_df["ticker"] = quotes_df["ticker"].astype(str)

    # sort both sides
    exec_df = exec_df.sort_values(
        by=["order_time", "symbol"],
        kind="mergesort"
    ).reset_index(drop=True)

    quotes_df = quotes_df.sort_values(
        by=["sip_timestamp", "ticker"],
        kind="mergesort"
    ).reset_index(drop=True)

    # merge using order_time as the lookup time
    master = pd.merge_asof(
        exec_df,
        quotes_df,
        left_on="order_time",
        right_on="sip_timestamp",
        left_by="symbol",
        right_by="ticker",
        direction="backward"
    )

    master = master.drop(columns=["ticker", "sip_timestamp"])

    print("merging successful")

    return master


def add_price_improvement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price_improvement column to the dataframe.
    BUY (Side == 1):  limit - execution
    SELL           :  execution - limit
    """

    #df = df.dropna(subset=["bid_price", "ask_price"]).copy()

    is_buy = df["Side"] == 1

    df.loc[:, "price_improvement"] = np.where(
        is_buy,
        df["limit_price"] - df["execution_price"],   # BUY
        df["execution_price"] - df["limit_price"]    # SELL
    )

    return df


def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    
    df["side_num"] = np.where(df["Side"] == 1, 1, 0)

    features = [
        "side_num",
        "order_qty",
        "limit_price",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size"
    ]

    df_model = df[features + ["price_improvement", "exchange"]].copy()
    df_model = df_model.dropna()    # drop na before model training

    return df_model


def train_best_model_for_exchange(feature_df):
    """
    """
    MODEL_CANDIDATES = {
    "random_forest": RandomForestRegressor(),
    "linear_regression": LinearRegression(),
    "ridge": Ridge(),
    "lasso": Lasso(),
    #"gradient_boosting": GradientBoostingRegressor(),
    #"xgboost": xgb.XGBRegressor(),
    #"lightgbm": lgb.LGBMRegressor()
}

    PARAM_GRIDS = {
    "random_forest": {        
        "model__n_estimators": [50, 100, 200, 400],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [4, 8, 16],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.8],
        "model__bootstrap": [True],
    },
    "linear_regression": {},
    "ridge": {
        "model__alpha": [0.01, 0.1, 1.0, 10, 100]
    },
    "lasso": {
        "model__alpha": [0.001, 0.01, 0.1, 10, 100]
    },
    "gradient_boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1]
    }
}

    X = feature_df.drop(columns=["price_improvement", "exchange"])
    y = feature_df["price_improvement"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_score = -np.inf
    best_model = None
    best_name = None
    best_rmse = None

    for model_name, model in MODEL_CANDIDATES.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        param_grid = PARAM_GRIDS.get(model_name, {})

        if param_grid:
            grid = GridSearchCV(
                pipe,
                param_grid,
                cv=5,
                scoring="r2",
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            candidate = grid.best_estimator_
        else: # linear regression does not have grid
            pipe.fit(X_train, y_train)
            candidate = pipe

        # Evaluate
        y_pred = candidate.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"{model_name}: R²={r2:.4f}, RMSE={rmse:.4f}")

        if r2 > best_score:
            best_score = r2
            best_model = candidate
            best_name = model_name
            best_rmse = rmse

    return best_model, best_name, best_score, best_rmse


def train_models_all_exchanges(feature_df):
    models = {}
    exchanges = feature_df["exchange"].unique()

    for exchange in tqdm(exchanges, desc="Exchanges"):
        df_exchange = feature_df[feature_df["exchange"] == exchange]

        # skip exchanges without enough data
        if len(df_exchange) < 100:
            continue

        model_tuple = train_best_model_for_exchange(df_exchange)
        models[exchange] = model_tuple[0] # recall that train function above 
                                          # returns many things including performance metrics
                                          # hence [0] selects the model itself
        print(f"Model {model_tuple[0]} trained")

    return models


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--exec_path", required=True, help="Path to executions.csv")
    parser.add_argument("--quotes_path", required=True, help="Path to quotes file")
    args = parser.parse_args()

    # read csv
    df_exec = read_executions(args.exec_path)
    df_quotes = read_quotes(args.quotes_path)

    # creating master dataframe
    master = merge_execution_and_quotes(df_exec, df_quotes)
    master = add_price_improvement(master)

    # build feature dataframe
    feature_df = build_feature_df(master)

    # train model
    models = train_models_all_exchanges(feature_df)

    # Save models
    for exchange_id, model in models.items():
        joblib.dump(model, f"models/{exchange_id}.pkl")
        print(f"Model {exchange_id} saved")

if __name__ == "__main__":
    main()