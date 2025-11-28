"""

This is the main file for model training.

Usage: python train_model.py 
    --exec_path /opt/assignment3/executions.csv
    --quotes_path /opt/assignment4/quotes_2025-09-10_small.csv.gz


"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, \
                             GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

import argparse
import joblib
from tqdm import tqdm


from build_dataframe_utils import (
    read_executions,
    read_quotes,
    merge_execution_and_quotes,
    add_price_improvement,
    build_feature_df,
)

MODEL_CANDIDATES = {
    "random_forest": RandomForestRegressor(),
    "linear_regression": LinearRegression(),
    "ridge": Ridge(),
    "lasso": Lasso(),
    "gradient_boosting": GradientBoostingRegressor(),
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


def train_model_single_exchange(feature_df: pd.DataFrame):
    """
    Train regression models for a single exchange and select the best one.

    This function evaluates several candidate models (Random Forest, Linear Regression,
    Ridge, Lasso). For models with hyperparameters, it performs a GridSearchCV using
    R² as the scoring metric. The best-performing model on the validation set is returned.

    feature_df (pd.DataFrame):
        A DataFrame containing all observations for one exchange.
        df include:
            - predictors: side_num, order_qty, limit_price, bid_price,
                          ask_price, bid_size, ask_size
            - target: price_improvement
            - column 'exchange' (will be dropped)

    Returns:
        best_model (sklearn.base.BaseEstimator):
            The fitted sklearn Pipeline model that achieved the highest R².
    """


    X = feature_df.drop(columns=["price_improvement", "exchange"])
    y = feature_df["price_improvement"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_score = -np.inf
    best_model = None

    for model_name, model in MODEL_CANDIDATES.items():
        
        print(f"Training {model_name} Initiated")
        
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

    return best_model


def train_models_all_exchanges(
        feature_df: pd.DataFrame, 
        min_rows: int = 100
        ):

    """
    Use the train_model_single_exchange function and 
    train one model per exchange and 
    return a dictionary of fitted models.

    feature_df (pd.DataFrame):
        DataFrame containing features and target for all exchanges.
        include:
          - "exchange": exchange identifier
          - "price_improvement": target variable
          - All feature columns required by `train_model_single_exchange`.
    min_rows (int), optional:
        Minimum number of rows required for an exchange to be modeled.
        Exchanges with fewer than `min_rows` observations are skipped.
        Default is 100.

    Returns:
        best_model (sklearn estimator): 
            The trained model (pipeline) with highest validation R².
    """

    models = {}
    exchanges = feature_df["exchange"].unique()

    for exchange in tqdm(exchanges, desc="Exchanges"):
        df_exchange = feature_df[feature_df["exchange"] == exchange]

        # skip exchanges without enough data
        if len(df_exchange) < min_rows:
            continue

        model = train_model_single_exchange(df_exchange)
        models[exchange] = model

        print(f"/n Model {model} trained /n")

    return models


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training models on execution and quote data.

    Returns:
        argparse.Namespace
            Parsed arguments with the following attributes:
            - exec_path : str
                Path to executions CSV file.
            - quotes_path : str
                Path to quotes file.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--exec_path", 
                        required=True, 
                        help="Path to executions.csv")

    parser.add_argument("--quotes_path", 
                        required=True, 
                        help="Path to quotes file")

    return parser.parse_args()


def main():

    args = parse_args()

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