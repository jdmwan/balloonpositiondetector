import pandas as pd

df = pd.read_csv("BalloonDataset/labels_train.csv")

# Split into 80% train (first 400), 20% test (last 100)
train_df = df.iloc[:400]
test_df = df.iloc[400:]

train_df.to_csv("BalloonDataset/labels_train_split.csv", index=False)
test_df.to_csv("BalloonDataset/labels_test_split.csv", index=False)