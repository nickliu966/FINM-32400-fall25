import pandas as pd
import argparse
import sys


if len(sys.argv) < 5:
    print("Missing arguments!\nPlease provide the --input_fix_file followed by " \
    "the fix file name as well as --output_csv_file followed by the csv file name\n\n" \
    "Usage Example:\n" \
    "python fix_to_csv.py --input_fix_file cleaned.fix --output_csv_file output.csv\n")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Calculate execuition metrics from data of fix messgaes")
parser.add_argument("--input_csv_file", help="Input csv files that contains metrics of interests")
parser.add_argument("--output_metrics_file", help="Output csv files that contains avg execution speed" \
                                                  "and price difference per exchange")
args = parser.parse_args()

df = pd.read_csv(args.input_csv_file)

df["ExecutionTransactTime"] = pd.to_datetime(df["ExecutionTransactTime"])
df["OrderTransactTime"] = pd.to_datetime(df["OrderTransactTime"])
df["exec_speed"] = (df["OrderTransactTime"] - df["ExecutionTransactTime"]).dt.total_seconds()

df["savings"] = abs(
    (df["LimitPrice"] - df["AvgPx"]) * df["OrderQty"]
    )

df_groupby_exchange = df.groupby("LastMkt")

metrics = (
        df_groupby_exchange[['exec_speed', 'savings']]
        .mean()
        .reset_index()
        .rename(columns={
            'exec_speed': 'AvgExecSpeedSecs',
            'savings': 'AvgPriceImprovement'
            })
        )

metrics.to_csv(args.output_metrics_file, index=False)