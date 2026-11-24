import json
from pathlib import Path
import pandas as pd

df = pd.read_csv("data/raw/creditcard.csv")
cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

fraud = df[df["Class"] == 1].iloc[3]
legit = df[df["Class"] == 0].sample(1, random_state=42).iloc[0]

out = Path("samples")
out.mkdir(exist_ok=True)

for name, row in [("fraud", fraud), ("legit", legit)]:
    data = {c: float(row[c]) for c in cols}
    with open(out / f"{name}.json", "w") as f:
        json.dump(data, f, indent=2)

print("Done: samples/fraud.json and samples/legit.json")