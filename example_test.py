import pandas as pd

df = pd.read_csv("hf://datasets/qppd/bank-transaction-fraud/dataset.csv")
#save data csv to local
df.to_csv("data/example_fraud.csv", index=False)
