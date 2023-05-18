#!/home/grev/code/oldcudaooba/installer_files/env python3
# reads lora_results_*.txt files and filters to lines containing the terms brendan, fraser and whale
# splits line according to | and prints in a table

import sys
import os
import re
import pandas as pd

# get the directory of this script
script_dir = os.path.dirname(os.path.realpath(__file__))

# glob for files lora_results_*.txt in script_dir
files = os.listdir(script_dir)
files = [f for f in files if re.match(r"lora_results.*\.txt", f)]

# read each file into a dataframe
dfs = []
for f in files:
    df = pd.read_csv(f, sep="|", names=["lorametrics", "prompt", "seed", "inference"])
    dfs.append(df)

# concatenate the dataframes
df = pd.concat(dfs)

# handle the case where the inference field is empty/NaN
df["inference"] = df["inference"].fillna("")
# filter to rows containing the terms brendan, fraser and whale
#df = df[df["inference"].str.contains("brendan|fraser|whale", case=False)]
df = df[df["inference"].str.contains("brendan|fraser|whale|force", case=False)]

# remove duplicates on the lorametrics column
#df = df.drop_duplicates(subset=["lorametrics","inference"])

# 200-128-256-256-1-1e-05
# split lorametrics 5 times on - and expand into columns
# avoid error ValueError: Columns must be same length as key
df[["epochs", "r", "alpha", "cutoff", "warmup", "lr"]] = df["lorametrics"].str.split("-", expand=True, n=5)

# drop the lorametrics column
#df = df.drop(columns=["lorametrics"])

# convert the columns to numeric
df[["epochs", "r", "alpha", "cutoff", "warmup", "lr"]] = df[["epochs", "r", "alpha", "cutoff", "warmup", "lr"]].apply(pd.to_numeric)

# sort the dataframe by the columns
df = df.sort_values(by=["lr","r", "alpha", "cutoff", "epochs", "warmup"])

# print the dataframe
print(df)

#export to csv
df.to_csv("lora_results.csv", index=False)
