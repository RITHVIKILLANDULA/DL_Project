from __future__ import annotations



import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fake_news.data.io import build_unified_dataset
from fake_news.data.split import split_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--liar-path", type=Path, required=True)
    parser.add_argument("--fakenewsnet-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = build_unified_dataset(args.liar_path, args.fakenewsnet_path)
    split = split_dataframe(df)

    split.train.to_csv(args.out_dir / "train.csv", index=False)
    split.val.to_csv(args.out_dir / "val.csv", index=False)
    split.test.to_csv(args.out_dir / "test.csv", index=False)

    print(f"Saved dataset splits to {args.out_dir.resolve()}")
    print(f"Train={len(split.train)} Val={len(split.val)} Test={len(split.test)}")


if __name__ == "__main__":
    main()

