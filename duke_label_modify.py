import pandas as pd

df = pd.read_csv("./data/modified_duke.csv")
df = df[df["label"] != 2]
df.to_csv("./data/modified_duke.csv", index=False)
