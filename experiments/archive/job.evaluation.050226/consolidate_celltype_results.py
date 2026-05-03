import argparse
import os
import pandas as pd
from pathlib import Path


def main(args):
    CONFIG_FILE = args.config_file
    DATASET_NAME = args.dataset_name
    BASE_AURC_DIR = args.base_aurc_dir
    OUTPUT_DIR = args.output_dir
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read cell type configuration
    config_df = pd.read_csv(CONFIG_FILE,index_col=0)
    config_df = config_df.reset_index(names='celltype')
    celltypes = config_df['celltype'].tolist()
    
    print(f"Consolidating results for {len(celltypes)} cell types: {celltypes}")
    
    # Dictionary to hold concatenated results by ground truth source
    consolidated = {}
    
    for celltype in celltypes:
        celltype_dir = os.path.join(BASE_AURC_DIR, celltype)
        
        if not os.path.exists(celltype_dir):
            print(f"Warning: Directory not found for {celltype}: {celltype_dir}")
            continue
        
        # Find all AUERC result files
        for file in os.listdir(celltype_dir):
            if not file.endswith('_auerc.csv'):
                continue
            
            # Extract ground truth source (e.g., "gtex", "onek1k", "abc", etc.)
            # File format: {source}_{dataset}_{celltype}_auerc.csv
            # e.g., onek1k_bmmc_nk_auerc.csv -> source = onek1k
            parts = file.replace('_auerc.csv', '').split('_')
            
            # Heuristic: first part is usually the source
            source = parts[0]
            
            filepath = os.path.join(celltype_dir, file)
            df = pd.read_csv(filepath)
            
            # Add celltype column
            df['celltype'] = celltype
            
            # Add to consolidated dictionary
            if source not in consolidated:
                consolidated[source] = []
            consolidated[source].append(df)
            
            print(f"  Added {celltype}/{file} -> source={source}")
    
    # Concatenate and save results by source
    for source, dfs in consolidated.items():
        if not dfs:
            continue
        
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Reorder columns to put celltype first
        cols = ['celltype'] + [c for c in full_df.columns if c != 'celltype']
        full_df = full_df[cols]
        
        output_file = os.path.join(OUTPUT_DIR, f'{source}_{DATASET_NAME}_all_celltypes_auerc.csv')
        full_df.to_csv(output_file, index=False)
        
        print(f"\nConsolidated {len(dfs)} cell types for {source}:")
        print(f"  Shape: {full_df.shape}")
        print(f"  Saved to: {output_file}")
        print(f"  Cell types: {full_df['celltype'].unique().tolist()}")
    
    print("\nConsolidation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidate cell-type-specific AUERC results into unified tables"
    )
    
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to celltype configuration CSV"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (e.g., bmmc)"
    )
    parser.add_argument(
        "--base_aurc_dir",
        type=str,
        required=True,
        help="Base directory containing cell-type-specific AURC results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for consolidated results"
    )
    
    args = parser.parse_args()
    main(args)