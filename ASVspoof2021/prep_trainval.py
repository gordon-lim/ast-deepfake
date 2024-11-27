import os
import pandas as pd
from sklearn.model_selection import train_test_split

print("Start preparing data for ASVspoof...")

# Load the CSV file
base_path = os.getcwd()
datafile_path = os.path.join(base_path, 'datafiles')

if not os.path.exists(datafile_path):
    os.mkdir(datafile_path)

file_path = os.path.join(base_path, 'train_la_data.csv')
df = pd.read_csv(file_path)

# Perform stratified train-validation split
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['Label'],
    random_state=42
)

# Save train and validation splits to CSV
train_csv_path = os.path.join(datafile_path, 'la_train.csv')
val_csv_path = os.path.join(datafile_path, 'la_val.csv')

train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)

print(f'Train samples: {len(train_df)}, Validation samples: {len(val_df)}')
print('Finished ASVspoof Preparation')