from datasets import load_dataset
import pandas as pd
from pathlib import Path

Path("data").mkdir(exist_ok=True)

ds = load_dataset("SetFit/enron_spam")
df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)
df.to_csv("data/enron_spam_data.csv", index=False)

print(df.shape)  # expect (33716, 7)
print(df["label"].value_counts())