from pathlib import Path
import pandas as pd

def load_data(data_dir: str | Path = "data"):
    data_dir = Path(data_dir)
    deliveries = pd.read_csv(data_dir / "deliveries.csv")
    matches = pd.read_csv(data_dir / "matches.csv")
    return deliveries, matches
