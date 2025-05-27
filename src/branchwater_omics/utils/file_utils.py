# ===================================== IMPORTS ====================================== #

from collections import defaultdict
import os
import re
from typing import List, Dict

import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# ========================== INITIALIZATION & CONFIGURATION ========================== #


# ==================================== FUNCTIONS ===================================== #

def get_files_in_dir(dir_path: str) -> List[str]:
    """
    Get list of file names in a directory.
    
    Args:
        dir_path: Path to directory to scan
        
    Returns:
        List of filenames (excluding directories) in the specified directory
    """
    with os.scandir(dir_path) as entries:
        return [entry.name for entry in entries if entry.is_file()]
        

def get_dirs_in_dir(dir_path: str) -> List[str]:
    """
    Get list of directory names in a directory.
    
    Args:
        dir_path: Path to directory to scan
        
    Returns:
        List of subdirectory names (excluding files) in the specified directory
    """
    with os.scandir(dir_path) as entries:
        return [entry.name for entry in entries if entry.is_dir()]
        

def group_mmseqs_and_wdir_files(
  motupan_90_mmseqs_dir: str, 
  motupan_90_wdir_dir: str
) -> Dict[str, Dict]:
    """
    Group related MMseqs and Motupan result files by accession number.
    
    Processes two input directories containing different file types:
    - MMseqs results (.m8 files)
    - Motupan workflow files (.json files)
    
    Groups files by accession number using filename patterns and combines metadata.
    
    Args:
        motupan_90_mmseqs_dir: Path to directory containing MMseqs .m8 result files
        motupan_90_wdir_dir: Path to directory containing Motupan JSON result files
        
    Returns:
        Dictionary grouped by accession numbers with structure:
        {
            accession: {
                "m8_file": path_to.m8,          # Path to MMseqs result file
                "mmseqs90_json": path_to.json,  # Path to MMseqs JSON file
                "motupan_json": path_to.json,   # Path to Motupan JSON file
                "species_identifier": str,      # Species ID from .m8 filename
                "species_name": str,            # Species name from JSON files
                "db": str,                     # Database type (RS/GB)
                "accession": str                # Accession number
            }
        }
    """
    # Regex patterns for parsing filenames in each directory
    # Pattern for MMseqs directory: s__<species>_sp<ID>--GB_<accession>.m8
    pattern_motupan_90_mmseqs_dir = re.compile(
        r"^s__([A-Za-z0-9_.-]+)_sp(\d+)--GB_(GCA_\d+\.\d+)\.m8$"
    )
    
    # Pattern for workflow directory: s__<species>--<DB>_<accession>.<tool>.json
    pattern_motupan_90_wdir_dir = re.compile(
        r"^s__([A-Za-z0-9_.-]+)--(RS|GB)_(GC[AF]_\d+\.\d+)\.(mmseqs90|motupan)\.json$"
    )

    # Initialize grouped file structure with default values
    file_groups = defaultdict(lambda: {
        "m8_file": None,
        "mmseqs90_json": None,
        "motupan_json": None,
        "species_identifier": None,  # Format: species_prefix_spID from .m8 files
        "species_name": None,        # Full species name from JSON files
        "db": None,                  # Database source (RS=RefSeq, GB=GenBank)
        "accession": None            # GenBank/RefSeq accession number
    })

    def parse_and_group(directory: str):
        """Helper function to process files in a directory and group them by accession."""
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            # Process MMseqs directory files (.m8)
            if directory == motupan_90_mmseqs_dir:
                match = pattern_motupan_90_mmseqs_dir.match(filename)
                if match:
                    species_prefix, sp_id, gca = match.groups()
                    # Normalize accession format (GCA_XXXX.XX)
                    accession = f"GCA_{gca.split('_')[-1]}" if "GCA_" not in gca else gca
                    
                    # Update group with MMseqs file information
                    group = file_groups[accession]
                    group["m8_file"] = filepath
                    group["species_identifier"] = f"{species_prefix}_sp{sp_id}"
                    group["db"] = "GB"
                    group["accession"] = accession

            # Process workflow directory files (.json)
            elif directory == motupan_90_wdir_dir:
                match = pattern_motupan_90_wdir_dir.match(filename)
                if match:
                    species_name, db, accession, tool = match.groups()
                    group = file_groups[accession]
                    
                    # Update common metadata
                    group["species_name"] = species_name
                    group["db"] = db
                    group["accession"] = accession
                    
                    # Update tool-specific paths
                    if tool == "mmseqs90":
                        group["mmseqs90_json"] = filepath
                    elif tool == "motupan":
                        group["motupan_json"] = filepath

    # Process both input directories
    parse_and_group(motupan_90_mmseqs_dir)
    parse_and_group(motupan_90_wdir_dir)

    # Remove empty groups that have no associated files
    file_groups = {k: v for k, v in file_groups.items() if any(v.values())}
    
    return file_groups
    

def load_grouped_data(
    grouped_data,
    read_m8: bool = True,
    read_json: bool = True,
    use_progress: bool = True
) -> List[Dict]:
    """
    Load and consolidate data from grouped file paths into structured data entries.
    
    Processes file groups created by group_mmseqs_and_wdir_files() to load:
    - MMseqs tabular data (.m8 files)
    - MMseqs JSON results
    - Motupan JSON results
    Handles errors gracefully while maintaining progress visualization.

    Args:
        grouped_data: Input data structure from group_mmseqs_and_wdir_files(),
                      either as a dict (accession-keyed) or list of group dicts
        read_m8: Whether to load MMseqs .m8 tabular data (default: True)
        read_json: Whether to load JSON files (both MMseqs and Motupan) (default: True)
        use_progress: Show progress bar with accession tracking (default: True)

    Returns:
        List of dictionaries containing structured data, each with:
        - metadata: Accession information and species identifiers
        - m8_data: DataFrame with MMseqs alignment results (if loaded)
        - mmseqs90_data: MMseqs JSON analysis results (if loaded)
        - motupan_data: Motupan JSON results (nested structure simplified)

    Raises:
        FileNotFoundError: If referenced files in group_data don't exist
        JSONDecodeError: If JSON files contain invalid syntax (captured and reported)
        pd.errors.ParserError: For malformed .m8 files (captured and reported)

    Example:
        >>> grouped = group_mmseqs_and_wdir_files("mmseqs_dir", "wdir_dir")
        >>> loaded = load_grouped_data(grouped, read_m8=True, read_json=True)
        >>> first_entry = loaded[0]
        >>> print(first_entry['metadata']['accession'])
        'GCA_123456.1'
        >>> print(first_entry['m8_data'].columns)
        Index(['query_id', 'target_id', 'identity', ...], dtype='object')
    """
    if isinstance(grouped_data, dict):
        grouped_data = list(grouped_data.values())
    
    loaded_data = []
    
    with tqdm(
        total=len(grouped_data),
        desc="Loading data",
        disable=not use_progress,
        unit="group",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ) as pbar:
        for group in grouped_data:
            current_accession = group.get("accession", "unknown")
            pbar.set_postfix_str(f"ACC: {current_accession[:15]}")  # Show truncated accession
            
            entry = {
                "metadata": {
                    "accession": current_accession,
                    "species_identifier": group.get("species_identifier"),
                    "species_name": group.get("species_name"),
                    "db": group.get("db"),
                },
                "m8_data": None,
                "mmseqs90_data": None,
                "motupan_data": None
            }
            
            # Load .m8 file (silent on success)
            if read_m8 and group.get("m8_file"):
                try:
                    entry["m8_data"] = pd.read_csv(
                        group["m8_file"], 
                        sep="\t", 
                        header=None,
                        comment='#',
                        names=[
                            "query_id", "target_id", "identity", "alignment_length",
                            "mismatches", "gap_openings", "q_start", "q_end",
                            "t_start", "t_end", "e_value", "bit_score"
                        ]
                    )
                except Exception as e:
                    tqdm.write(f"\nError in {group['m8_file']}: {str(e)}")
            
            # Load JSON files (silent on success)
            if read_json:
                for tool in ["mmseqs90", "motupan"]:
                    json_path = group.get(f"{tool}_json")
                    if json_path:
                        try:
                            with open(json_path, "r") as f:
                                data = json.load(f)
                                # For "motupan", extract the first nested value from the JSON
                                if tool == "motupan":
                                    entry[f"{tool}_data"] = next(iter(data.values()))
                                else:
                                    entry[f"{tool}_data"] = data
                        except Exception as e:
                            tqdm.write(f"\nError in {json_path}: {str(e)}")
            
            loaded_data.append(entry)
            pbar.update(1)
    
    return loaded_data


def build_and_parse_file_dict(directory, sub_dirs=None, files=None, parsed=True):
    """
    Recursively builds and parses files in a directory structure.
    Returns a nested dict: { sub_dir: { filename: DataFrame|dict|path } }.
    Shows one accurate tqdm bar for just the matching files.
    """
    # defaults
    sub_dirs = sub_dirs or ['viruses', 'plasmids']
    files = files or [
        'all_matched_clusters_across_clade_df.csv',
        'aux_core_cluster_matches_table.csv',
        'geNomad_ids_across_clade_df.csv',
        'results_dict_original_format.npy'
    ]
    sub_dirs_set = set(sub_dirs)

    # Phase 1: count how many matching files there are
    total = sum(
        1
        for root, _, filenames in os.walk(directory)
        for f in filenames
        if os.path.basename(root) in sub_dirs_set and f in files
    )

    file_dict = defaultdict(dict)
    pbar = tqdm(total=total, desc="Processing files")

    # Phase 2: walk & parse
    for root, _, filenames in os.walk(directory):
        category = os.path.basename(root)
        if category not in sub_dirs_set:
            continue

        for fname in filenames:
            if fname not in files:
                continue

            path = os.path.join(root, fname)

            if parsed:
                if fname.endswith('.csv'):
                    file_dict[category][fname] = pd.read_csv(path)
                elif fname.endswith('.npy'):
                    file_dict[category][fname] = np.load(path, allow_pickle=True).item()
            else:
                file_dict[category][fname] = path

            pbar.update(1)

    pbar.close()
    return dict(file_dict)


import pandas as pd
import json
import psutil
import os
from typing import List, Dict, Generator
from functools import wraps
from itertools import islice

def monitor_memory(func):
    """Decorator to track memory usage during processing"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss
        result = func(*args, **kwargs)
        end_mem = process.memory_info().rss
        print(f"Memory usage for {func.__name__}:")
        print(f"  Start: {start_mem / 1024**2:.1f}MB")
        print(f"  End: {end_mem / 1024**2:.1f}MB")
        print(f"  Delta: {(end_mem - start_mem)/1024**2:.1f}MB")
        return result
    return wrapper

def chunk_processor(chunk_size: int = 100):
    """Decorator factory for chunked DataFrame processing"""
    def decorator(func):
        @wraps(func)
        def wrapper(loaded_data, *args, **kwargs):
            iterator = iter(loaded_data)
            final_dfs = []
            
            while True:
                chunk = list(islice(iterator, chunk_size))
                if not chunk:
                    break
                
                print(f"\nProcessing chunk {len(final_dfs) + 1}")
                chunk_result = func(chunk, *args, **kwargs)
                final_dfs.append(chunk_result)
                
                # Explicit memory cleanup
                del chunk
                if 'chunk_result' in locals():
                    del chunk_result
                    
            return pd.concat(final_dfs, ignore_index=True) if final_dfs else pd.DataFrame()
        return wrapper
    return decorator

@monitor_memory
@chunk_processor(chunk_size=100)  # Adjust chunk_size based on system memory
def combine_loaded_data(loaded_data: List[Dict]) -> pd.DataFrame:
    """Memory-optimized chunk processor with bulk operations"""
    chunk_dfs = []
    
    for entry in loaded_data:
        if not entry.get("m8_data") or entry["m8_data"].empty:
            continue

        # Base DataFrame
        df = entry["m8_data"]
        
        # Prepare metadata columns
        metadata = entry["metadata"]
        meta_cols = {k: [v] * len(df) for k, v in metadata.items()}
        
        # Process JSON data with type handling
        json_data = {}
        for tool in ["mmseqs90", "motupan"]:
            data = entry.get(f"{tool}_data")
            if data:
                try:
                    flat = pd.json_normalize(data, sep="_").iloc[0].to_dict()
                    json_data.update({
                        f"{tool}_{k}": json.dumps(v) if isinstance(v, (list, dict)) else v
                        for k, v in flat.items()
                    })
                except Exception:
                    pass
                    
        json_cols = {k: [v] * len(df) for k, v in json_data.items()}
        
        # Bulk column creation
        new_cols_df = pd.DataFrame({**meta_cols, **json_cols}, index=df.index)
        
        # Merge with concat to avoid fragmentation
        merged = pd.concat([df, new_cols_df], axis=1)
        
        # Optimize dtypes
        merged = merged.astype({
            col: "category" for col in meta_cols.keys()
            if merged[col].nunique() / len(merged) < 0.5
        })
        
        # Downcast numerics immediately
        num_cols = merged.select_dtypes(include='number').columns
        merged[num_cols] = merged[num_cols].apply(pd.to_numeric, downcast='unsigned')
        
        chunk_dfs.append(merged)
    
    return pd.concat(chunk_dfs, ignore_index=True) if chunk_dfs else pd.DataFrame()
