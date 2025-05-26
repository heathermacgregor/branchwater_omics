# ===================================== IMPORTS ====================================== #

import os
import re
from collections import defaultdict
import json
import pandas as pd
from typing import List, Dict
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
