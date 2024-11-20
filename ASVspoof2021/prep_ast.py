import numpy as np
import json
import os
import zipfile
import wget
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


print("start preparing data for asvspoof...")

# Load the CSV file
base_path = os.getcwd()

datafile_path = base_path + '/datafiles'
if os.path.exists(datafile_path) == False:
    os.mkdir(datafile_path)

file_path = base_path + '/combined_df_data.csv'
df = pd.read_csv(file_path)

n_splits = 5

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

df['fold'] = -1

# Assign each sample to a fold, stratifying by the 'Label' column
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['Label'])):
    df.loc[val_idx, 'fold'] = fold


# Iterate over each fold to create train-eval splits
for fold in range(n_splits):
    # Filter rows for the current fold
    eval_df = df[df['fold'] == fold]
    train_df = df[df['fold'] != fold]

    # Perform stratified train-test split on the training set to ensure 8:2 ratio
    train_df, eval_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df['Label'],
        random_state=42
    )

    train_df = train_df.drop(columns=['fold'])
    eval_df = eval_df.drop(columns=['fold'])

    # Save train and eval splits to CSV
    train_csv_path = datafile_path + f'/df_train_fold_{fold+1}.csv'
    eval_csv_path = datafile_path + f'/df_eval_fold_{fold+1}.csv'
    train_df.to_csv(train_csv_path, index=False)
    eval_df.to_csv(eval_csv_path, index=False)

    print(
        f'Fold {fold+1}: Train samples - {len(train_df)}, Eval samples - {len(eval_df)}')


print('Finished asvspoof2021 Preparation')
