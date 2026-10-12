from pathlib import Path
import hashlib
import json
import os
os.environ.pop("SSLKEYLOGFILE", None)

print("SSLKEYLOGFILE =", os.environ.get("SSLKEYLOGFILE"))

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "mrpc"


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)

    return sha.hexdigest()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MRPC from Hugging Face...")
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
    )

    print(dataset)

    expected_columns = {
        "sentence1",
        "sentence2",
        "label",
        "idx",
    }

    metadata = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "label_mapping": {
            "0": "not_paraphrase",
            "1": "paraphrase",
        },
        "splits": {},
    }

    for split_name in ["train", "validation", "test"]:

        df = dataset[split_name].to_pandas()

        missing = expected_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{split_name} is missing columns: {missing}"
            )

        # Keep only the columns every team member needs.
        df = df[
            [
                "idx",
                "sentence1",
                "sentence2",
                "label",
            ]
        ].copy()

        # IMPORTANT:
        # Do not shuffle.
        # Do not create new splits.
        # Do not clean the text here.
        # Preserve the official source dataset.

        output_path = RAW_DIR / f"{split_name}.csv"

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
        )

        metadata["splits"][split_name] = {
            "rows": len(df),
            "file": str(
                output_path.relative_to(PROJECT_ROOT)
            ),
            "sha256": sha256_file(output_path),
        }

        print(
            f"{split_name:10s}: "
            f"{len(df):5d} rows -> {output_path}"
        )

    metadata_path = DATA_DIR / "metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\nDataset prepared successfully.")
    print(f"Metadata: {metadata_path}")

    print("\nChecksums:")
    for split, info in metadata["splits"].items():
        print(
            split,
            info["sha256"],
        )


if __name__ == "__main__":
    main()