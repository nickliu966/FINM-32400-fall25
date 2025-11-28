"""
Utility file that contains the functions for preprocessing 

The functions reads the execution and nbbo files, merge, 
add price improvement, and ultimately builds feature dataframe as
training input.

The functions are imported in train_model.py

Note: I am using the older version of the execution.csv
"""

import pandas as pd
import numpy as np


def read_executions(path: str) -> pd.DataFrame:

    """
    Load and clean execution data from CSV.

    Input (str):
        Path to executions.csv.

    Returns (pd.Dataframe): 
        a dataframe that includes execution information
    """

    pd.DataFrame
    exec_df = pd.read_csv(path)

    # Convert timestamps
    exec_df['OrderTransactTime'] = pd.to_datetime(
        exec_df['OrderTransactTime'], 
        format='%Y%m%d-%H:%M:%S.%f'
    )
    exec_df['ExecutionTransactTime'] = pd.to_datetime(
        exec_df['ExecutionTransactTime'], 
        format='%Y%m%d-%H:%M:%S.%f'
    )

    # column names
    exec_df = exec_df.rename(columns={
        "OrderTransactTime": "order_time",
        "ExecutionTransactTime": "execution_time",
        "AvgPx": "execution_price",
        "LastMkt": "exchange",
        "Symbol": "symbol",
        "OrderQty": "order_qty",
        "LimitPrice": "limit_price"
    })

    # Make symbol categorical
    exec_df['symbol'] = exec_df['symbol'].astype('category')

    # Side as string ('1' = buy)
    exec_df['side'] = exec_df['Side'].astype(str)

    # Filter to market hours: 09:30–16:00
    market_open = pd.to_datetime("09:30").time()
    market_close = pd.to_datetime("16:00").time()

    exec_df = exec_df[
        (exec_df['order_time'].dt.time >= market_open) &
        (exec_df['order_time'].dt.time <= market_close)
    ]

    print("read execution successful")

    return exec_df


def read_quotes(path: str) -> pd.DataFrame:

    """
    Load and clean NBBO quote data.
    
    Input (str):
        Path to NBBO quotes gzip CSV.

    Returns (pd.DataFrame):
        a cleaned quote dataframe
    """

    quotes_df = pd.read_csv(path, compression="gzip")

    # sip_timestamp convert to datetime
    quotes_df['sip_timestamp'] = pd.to_datetime(quotes_df['sip_timestamp'], unit='ns')

    # Convert ticker to category
    quotes_df['ticker'] = quotes_df['ticker'].astype('category')

    # Filter market hours
    market_open = pd.to_datetime("09:30").time()
    market_close = pd.to_datetime("16:00").time()

    quotes_df = quotes_df[
        (quotes_df['sip_timestamp'].dt.time >= market_open) &
        (quotes_df['sip_timestamp'].dt.time <= market_close)
    ]

    print("read quotes successfule")

    return quotes_df


def merge_execution_and_quotes(exec_df: pd.DataFrame, 
                               quotes_df: pd.DataFrame
                               ) -> pd.DataFrame:
    """
    Merge execution data with NBBO quotes on timestamp.

    Input:
        exec_df (pd.DataFrame):
            Execution dataframe.

        quotes_df (pd.DataFrame):
            NBBO quotes dataframe.

    Returns (pd.DataFrame):
        A merged dataframe containing nearest NBBO before order_time.
    """

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
    master_df = pd.merge_asof(
        exec_df,
        quotes_df,
        left_on="order_time",
        right_on="sip_timestamp",
        left_by="symbol",
        right_by="ticker",
        direction="backward"
    )

    master_df = master_df.drop(columns=["ticker", "sip_timestamp"])

    print("merging successful")

    return master_df


def add_price_improvement(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price improvement.

    BUY: limit_price - execution_price  
    SELL: execution_price - limit_price

    Input:
        master_df (pd.DataFrame): Merged executions + NBBO dataframe.

    Returns:
        master_df (pd.DataFrame): A master dataFrame with new column 
                                  price_improvement.
    """

    #df = df.dropna(subset=["bid_price", "ask_price"]).copy()

    is_buy = master_df["Side"] == 1

    master_df.loc[:, "price_improvement"] = np.where(
        is_buy,
        master_df["limit_price"] - master_df["execution_price"],   # BUY
        master_df["execution_price"] - master_df["limit_price"]    # SELL
    )

    return master_df


def build_feature_df(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the master dataframe and transform into 
    a ML-ready feature dataframe.

    Input:
        master_df (pd.DataFrame):
        Full merged dataframe with price improvement.

    Returns:
        model_df (pd.DataFrame):
        A feature dataframe that contains the target and predictor variables
    """

    master_df["side_num"] = np.where(master_df["Side"] == 1, 1, 0)

    features = [
        "side_num",
        "order_qty",
        "limit_price",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size"
    ]

    model_df = master_df[features + ["price_improvement", "exchange"]].copy()
    model_df = model_df.dropna()    # drop na before model training

    return model_df