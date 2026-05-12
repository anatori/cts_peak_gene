import argparse
import os
import pandas as pd



def main(args):
    DATASET_PATH = args.dataset_path
    DATASETS = [m.strip() for m in args.datasets.split(",")]
    UNION_BED = args.union_bed

    union_bed = []
    for dataset in DATASETS:
        print(f"Processing dataset: {dataset}")
        dataset_path = os.path.join(DATASET_PATH, f"{dataset}.bed")
        if not os.path.isfile(dataset_path):
            print(f"Skipping: BED file not found for dataset '{dataset}' at expected path {dataset_path}")
            continue
        dataset_bed = pd.read_csv(dataset_path, sep="\t", header=None, names=["chr", "start", "end", "final_col"])
        print(f"Found BED file for dataset '{dataset}': {dataset_path} with {len(dataset_bed)} rows")
        dataset_bed["final_col"] = dataset_bed["final_col"].astype(str) + f";{dataset}"
        union_bed.append(dataset_bed)
    union_bed = pd.concat(union_bed, ignore_index=True)

    os.makedirs(os.path.dirname(UNION_BED), exist_ok=True)
    union_bed.to_csv(UNION_BED, header=False, index=False, sep="\t")
    print(f"Wrote {len(union_bed)} records to {UNION_BED}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="/ocean/projects/bio240051p/aprieto/bedfiles/validation/union_links")
    parser.add_argument("--union_bed", type=str, default="/ocean/projects/bio240051p/aprieto/bedfiles/analysis/multiome_union/multiome_union.bed")
    parser.add_argument("--datasets", type=str, default="bmmc,pbmc,brain,neat,shareseq")

    args = parser.parse_args()
    main(args)
