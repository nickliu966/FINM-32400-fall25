import re
import pandas as pd
import argparse
import csv

TAG_LOOKUP = {
    11: 'ClOrdID',
    60: 'ExecutionTransactTime',
    52: 'OrderTransactTime',
    55: 'Symbol',
    54: 'Side',
    38: 'OrderQty',
    44: 'LimitPrice',
    6: 'AvgPx',
    30: 'LastMkt',
    35: 'MsgType',
    150: 'ExecType',
    39: 'OrdStatus',
    40: 'OrdType'
}

def parse_fix(msg):


    # split transaction time from rest of fix that has tag=value
    pattern = r'(\d+-\d+:\d+:\d+\.\d+)\s:?\s(8=.*)'
    match = re.match(pattern, msg)

    time = match.group(1) # transaction time
    fix = match.group(2) # rest of fix message

    fix = fix.rstrip("\x01") # remove trailing soh

    fields = fix.split('\x01')

    parsed = {"time": time}

    for field in fields:
        tag, value = field.split("=", 1)
        tag_name = TAG_LOOKUP.get(int(tag), tag)
        parsed[tag_name] = value
    
    return parsed

parsed_messgaes = []

with open("cleaned.fix", "r") as f:
    for msg in f:
        parsed = parse_fix(msg)
        parsed_messgaes.append(parsed)


for msg in parsed_messgaes[:1]:
    print(msg)


fieldnames = list(TAG_LOOKUP.values())

with open("cleaned.fix", "r") as f, open("output.csv", "w", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for msg in f:
        parsed = parse_fix(msg)
        row = {col: parsed.get(col, '') for col in fieldnames}
        writer.writerow(row)