import numpy as np
import json
import os
import zipfile
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


# Paths and parameters
base_path = os.getcwd()
datafile_path = os.path.join(base_path, 'datafiles')
if not os.path.exists(datafile_path):
    os.mkdir(datafile_path)

file_path = os.path.join(base_path, 'train_la_data.csv')
df = pd.read_csv(file_path)


n_splits = 5

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Add a column for fold assignments
df['fold'] = -1

# Assign each sample to a fold, stratifying by 'Label'
for fold, (train_idx, eval_idx) in enumerate(skf.split(df, df['Label'])):
    df.loc[eval_idx, 'fold'] = fold

# Iterate through each fold
for fold in range(n_splits):
    # Evaluation data: the current fold
    eval_df = df[df['fold'] == fold]

    # Training data: all other folds
    train_df = df[df['fold'] != fold]

    # Drop the 'fold' column
    train_df = train_df.drop(columns=['fold'])
    eval_df = eval_df.drop(columns=['fold'])

    # Save train and eval splits to CSV
    train_csv_path = os.path.join(
        datafile_path, f'la_train_fold_{fold + 1}.csv')
    eval_csv_path = os.path.join(datafile_path, f'la_eval_fold_{fold + 1}.csv')

    train_df.to_csv(train_csv_path, index=False)
    eval_df.to_csv(eval_csv_path, index=False)

    print(
        f'Fold {fold + 1}: Train samples - {len(train_df)}, Eval samples - {len(eval_df)}')

print('Finished asvspoof2021 dataset prep.')
