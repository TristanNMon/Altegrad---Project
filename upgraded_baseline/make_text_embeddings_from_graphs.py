import os
import re
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def slugify_model_name(name: str) -> str:
    """
    Turn a model name like 'sentence-transformers/all-MiniLM-L6-v2'
    into a safe folder name like 'sentence-transformers_all-minilm-l6-v2'.
    """
    name = name.strip().lower()
    name = name.replace("/", "_")
    # keep only safe chars
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # Create folder
    out_dir = os.path.join("data", slugify_model_name(model_name))
    os.makedirs(out_dir, exist_ok=True)

    splits = ["train", "validation"]

    for split in splits:
        pkl = f"data/{split}_graphs.pkl"
        with open(pkl, "rb") as f:
            graphs = pickle.load(f)

        ids, descs = [], []
        for g in graphs:
            ids.append(getattr(g, "id", len(ids)))
            descs.append(getattr(g, "description", ""))

        embs = model.encode(
            descs, show_progress_bar=True, normalize_embeddings=True
        )  # (N, D)

        emb_strs = [",".join(f"{x:.6g}" for x in row) for row in embs]

        df = pd.DataFrame(
            {
                "ID": [str(i) for i in ids],  # baseline uses str(row["ID"])
                "embedding": emb_strs,  # comma-separated floats
            }
        )

        out = os.path.join(out_dir, f"{split}_text_embeddings.csv")
        df.to_csv(out, index=False)
        print(f"Saved {split} embeddings to {out}, shape={embs.shape}")


if __name__ == "__main__":
    main()
