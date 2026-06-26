import json
from pathlib import Path
import pandas as pd

def make_labels(data_dir, output_path):
    data_dir = Path(data_dir)

    classes = pd.read_csv(
        data_dir / "classes.txt",
        sep=" ",
        names=["class_id", "class_name"],
    )

    labels = {
        int(row.class_id) - 1: row.class_name.split(".", 1)[1].replace("_", " ")
        for row in classes.itertuples()
    }

    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)

    if __name__ == "__main__":
        make_labels(
            "data/raw/CUB_200_2011",
            "models/labels.json",
        )
