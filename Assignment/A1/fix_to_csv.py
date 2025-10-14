import re
import argparse
import csv
import sys

TAG_LOOKUP = {
    11: 'ClOrdID',
    60: 'ExecutionTransactTime',
    52: 'OrderTransactTime',
    55: 'Symbol',
    54: 'Side',
    38: 'OrderQty',
    44: 'LimitPrice',
    6: 'AvgPx',
    30: 'LastMkt'
}

def parse_fix(msg):
    """

    Input: (str) A single line of fix message

    Return: (dict) A dictionary that stores each of the tag/value pair 
    from the fix message

    """

    # split fix message time from rest of fix that has tag=value
    pattern = r'(\d+-\d+:\d+:\d+\.\d+)\s:?\s(8=.*)'
    match = re.match(pattern, msg)

    time = match.group(1) # fix message transact time
    fix = match.group(2) # rest of fix message

    fix = fix.rstrip("\x01") # remove trailing soh

    fields = fix.split('\x01') # extract tag-value pair

    parsed_msg = {"time": time} # create a dict to store value for later csv writing

    for field in fields:
        tag, value = field.split("=", 1)
        tag_name = TAG_LOOKUP.get(int(tag), tag)
        parsed_msg[tag_name] = value
    
    return parsed_msg


if len(sys.argv) < 5:
    print("Missing arguments!\nPlease provide the --input_fix_file followed by " \
    "the fix file name as well as --output_csv_file followed by the csv file name\n\n" \
    "Usage Example:\n" \
    "python fix_to_csv.py --input_fix_file cleaned.fix --output_csv_file output.csv\n")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Parse FIX messages into CSV format.")
parser.add_argument("--input_fix_file", help="Input FIX file.")
parser.add_argument("--output_csv_file", help="Output CSV file.")
args = parser.parse_args()


fieldnames = list(TAG_LOOKUP.values()) # find the tags of interest and input on these tags

with open(args.input_fix_file, "r") as f, open(args.output_csv_file, "w", newline='') as csvfile:

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for msg in f:
        parsed_msg = parse_fix(msg)
        row = {col: parsed_msg.get(col, '') for col in fieldnames}
        writer.writerow(row)