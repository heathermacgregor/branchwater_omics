# ===================================== IMPORTS ====================================== #

import sys
import os
import numpy as np
import pandas as pd

# ================================== LOCAL IMPORTS =================================== #

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from branchwater_omics.figures.pangenome import (
    completeness_vs_contamination, 
    contig_vs_scaffold,
    map
)
from branchwater_omics.utils.file_utils import (
    get_files_in_dir, 
    get_dirs_in_dir, 
    group_mmseqs_and_wdir_files, 
    load_grouped_data, 
    build_and_parse_file_dict
)
from branchwater_omics.utils.df_utils import (
    find_common_substrings, 
    parse_coordinate
)

# ================================= DEFAULT VALUES =================================== #

# Branchwater Results
branchwater_results_tsv = "/dvs_ro/cfs/cdirs/kbase/KE-common/data/all-gtdb-in-img.tsv"

# mOTUpan
motupan_90_dir = "/global/cfs/cdirs/kbase/ke_prototype/cjneely/KE-PANGENOMES/mOTUpan.90/"
motupan_90_wdir_dir = motupan_90_dir + "wdir/"
motupan_90_mmseqs_dir = motupan_90_dir + "mmseqs_results/"

# geNomad annotations
genomad_raw_results_dir = "/global/cfs/cdirs/kbase/ke_prototype/cjneely/KE-PANGENOMES/genomad-results"
genomad_parsed_results_dir = "/global/cfs/cdirs/kbase/ke_prototype/cjneely/KE-PANGENOMES/genomad-mapping-results"

# eggNOG annotations (done on e-value based MMseqs2 clusters)
eggnog_annotations_dir = "/global/cfs/cdirs/kbase/ke_prototype/cjneely/KE-PANGENOMES/eggnog-annotations"

# Pangenome Results
# - All the tables have headers. Note that the NCBI table is in a "long format" 
#   meaning that each entry of the data is a pair of the accession and an "attribute" 
#   of the data (e.g. country, collection_date, isolation_source, env_broad_scale). 
#   Then that pair will have a "content" associated with it (e.g. the attribute_name 
#    country for accession SAMEA867913 has the content of Malawi).
# - Some of these datasets were already filtered to our data of interest (within the 
#   set of 300k genomes I mentioned from GTDB r214.1). We are in the process of 
#   synchronizing and updating everything, but if you get to the point of querying 
#   for your genome/clade set and are missing significant matches please let us know
#   and we can discuss.
# - Please also note that the pangenome results Chris has linked use an updated 
#   protocol for running mOTUpan (90% similarity cutoff) versus the pangenome results 
#   in these "CDM tables" which use default similarity parameters.
pangenome_results_dir = "/global/cfs/cdirs/kbase/ke_prototype/mcashman/CDM/pangenome_data_tables"
pangenome_results_metadata_tsv = pangenome_results_dir + "/table_gtdb_metadata_V1.1.tsv"

# ==================================== FUNCTIONS ===================================== #

def get_branchwater_results(tsv_path):
    # Load the TSV file
    df = pd.read_csv(tsv_path, sep='\t', engine='pyarrow')
    return df


def get_motupan_results(mmseqs_dir_path, wdir_dir_path):
    mmseqs_dir_files = get_files_in_dir(mmseqs_dir_path)
    wdir_dir_files = get_files_in_dir(wdir_dir_path)
    
    
    file_groups = group_mmseqs_and_wdir_files(
      mmseqs_dir_path, wdir_dir_path
    )
    
    subset_file_groups = dict(list(file_groups.items())[:100])
    loaded_data = load_grouped_data(
        subset_file_groups.values(),
        read_m8=True,
        read_json=True,
        use_progress=True
    )
    return loaded_data
  

def get_genomad_results(raw_results_dir_path, parsed_results_dir_path):
    raw_results_dir_dirs = get_dirs_in_dir(raw_results_dir_path)
    parsed_results_dir_dirs = get_dirs_in_dir(parsed_results_dir_path)
    
    parsed_results_dict = build_and_parse_file_dict(genomad_parsed_results_dir)
    return parsed_results_dict
  

def get_eggnog_results(annotations_dir_path):
    annotations_dir_dirs = get_dirs_in_dir(annotations_dir)
    print(len(annotations_dir_dirs))
  

def get_pangenome_results(tsv_path):
    print(f"Loading pangenome results from '{tsv_path}'...")
    df = pd.read_csv(tsv_path, sep="\t", engine="c", low_memory=False, dtype_backend="pyarrow")
    print(df.shape)

    # Find top N biological terms (bigrams/trigrams)
    N = 30
    common_terms = find_common_substrings(
        df=df,
        column='ncbi_isolation_source',
        n_gram_range=(2, 3),  # Focus on 2-3 word phrases
        top_n=N,
        stopwords={'sample', 'isolated', 'from', 'of'},  # Custom stopwords
        min_term_length=4
    )
    print(common_terms)
    completeness_vs_contamination(df, '/global/homes/m/macgrego/figures/completeness_vs_contamination.png')

    # Filter High-Quality Genomes
    # Filter genomes meeting the MIMAG standard (e.g., completeness ≥ 90%, contamination ≤ 5%, strain heterogeneity ≤ 5%, and presence of rRNA/tRNA genes).
    high_quality = df[
        (df['checkm_completeness'] >= 90) &
        (df['checkm_contamination'] <= 5) &
        (df['checkm_strain_heterogeneity'] <= 5) &
        (df['ssu_count'] >= 1) &
        (df['trna_count'] >= 18)
    ].copy()

    # Parse Taxonomic Lineage
    # Filter for GTDB representative genomes (use 't'/'f' check)
    high_quality_reps = high_quality[high_quality['gtdb_representative'] == 't']
    
    # Split GTDB taxonomy into separate columns
    tax_ranks = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    df[tax_ranks] = df['gtdb_taxonomy'].str.split(';', expand=True)
    
    # Count genomes per phylum
    phylum_counts = df['phylum'].value_counts()
    print(phylum_counts)

    # Analyze rRNA and tRNA Genes
    # Identify genomes missing rRNA genes (critical for quality assessment)
    # Flag genomes with no SSU or LSU rRNA genes
    df['missing_rrna'] = (df['ssu_count'] == 0) | (df['lsu_23s_count'] == 0)
    
    # Summarize missing rRNA by taxonomic group
    missing_rrna_by_phylum = df.groupby('phylum')['missing_rrna'].mean()
    print(missing_rrna_by_phylum)

    # Compare Assembly Levels
    # Compare assembly statistics (e.g., n50_contigs) across NCBI assembly levels (Contig vs. Scaffold).
    # Convert to numeric (replace strings like "inf" with NaN)
    df_clean = df.copy()
    df_clean['n50_contigs'] = pd.to_numeric(df_clean['n50_contigs'], errors='coerce')
    
    # Replace infinite values with NaN and drop them
    df_clean['n50_contigs'] = df_clean['n50_contigs'].replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=['n50_contigs', 'ncbi_assembly_level'])
    
    # Convert to standard pandas/numpy dtype (not Arrow)
    df_clean['n50_contigs'] = df_clean['n50_contigs'].astype(float)
    
    # Optional: Filter valid assembly levels
    valid_levels = ['Contig', 'Scaffold', 'Complete Genome']
    df_clean = df_clean[df_clean['ncbi_assembly_level'].isin(valid_levels)]
    # Replace infinite values with NaN
    contig_vs_scaffold(
        df.replace([np.inf, -np.inf], np.nan, inplace=True), 
        valid_levels, 
        '/global/homes/m/macgrego/figures/contig_vs_scaffold.png'
    )

    # Geospatial Analysis
    # Map genomes by geographic location using ncbi_lat_lon.

    # Split into lat/lon using the parser
    df[['lat', 'lon']] = df['ncbi_lat_lon'].apply(
        lambda x: pd.Series(parse_coordinate(x))
    )

    # Drop rows with invalid coordinates
    df = df.dropna(subset=['lat', 'lon'])
    map(df, '/global/homes/m/macgrego/figures/map.png')
    return df
  

# ============================= DATAFRAME LINKAGE ANALYSIS =========================== #

def analyze_dataframe_linkage(dataframes, dataframe_names, sample_size=1000):
    """
    Analyze potential linkages between multiple DataFrames by checking:
    - Shared columns (potential join keys)
    - Data type compatibility
    - Value overlap in shared columns
    - Missing value percentages
    
    Args:
        dataframes (list): List of pandas DataFrames to analyze
        dataframe_names (list): Names corresponding to each DataFrame
        sample_size (int): Number of values to sample for overlap check
    
    Returns:
        dict: Analysis results with linkage potential
    """
    analysis_results = {}
    
    # Check all pairwise combinations
    for i, (df1, name1) in enumerate(zip(dataframes, dataframe_names)):
        for j, (df2, name2) in enumerate(zip(dataframes, dataframe_names)):
            if i >= j:  # Avoid duplicate checks and self-comparison
                continue
                
            common_cols = df1.columns.intersection(df2.columns)
            linkage_info = {}
            
            for col in common_cols:
                # Data type compatibility check
                dtype1 = df1[col].dtype
                dtype2 = df2[col].dtype
                dtype_compatible = pd.api.types.is_dtype_equal(dtype1, dtype2)
                
                # Sample values for overlap check
                sample1 = df1[col].dropna().sample(min(sample_size, len(df1))) if not df1.empty else pd.Series()
                sample2 = df2[col].dropna().sample(min(sample_size, len(df2))) if not df2.empty else pd.Series()
                
                # Calculate overlap metrics
                overlap = len(set(sample1).intersection(set(sample2))) if not sample1.empty and not sample2.empty else 0
                overlap_pct = overlap / max(len(sample1), 1) * 100  # Prevent division by zero
                
                # Missing values analysis
                missing_pct1 = df1[col].isna().mean() * 100
                missing_pct2 = df2[col].isna().mean() * 100
                
                linkage_info[col] = {
                    'dtypes': (str(dtype1), str(dtype2)),
                    'dtype_compatible': dtype_compatible,
                    'sample_overlap_pct': round(overlap_pct, 1),
                    'missing_pct_df1': round(missing_pct1, 1),
                    'missing_pct_df2': round(missing_pct2, 1),
                    'potential_key_strength': 'strong' if (overlap_pct > 75 and dtype_compatible) else 
                                            'moderate' if (overlap_pct > 50 and dtype_compatible) else 
                                            'weak'
                }
            
            if common_cols.any():
                analysis_results[f"{name1} <-> {name2}"] = linkage_info
                
    return analysis_results

def print_linkage_analysis(results):
    """Pretty-print the linkage analysis results"""
    for pair, linkages in results.items():
        print(f"\nLinkage analysis for: {pair}")
        for col, stats in linkages.items():
            print(f"  Column: {col}")
            print(f"  → Data types: {stats['dtypes'][0]} vs {stats['dtypes'][1]}")
            print(f"  → Type compatible: {'Yes' if stats['dtype_compatible'] else 'No'}")
            print(f"  → Sample overlap: {stats['sample_overlap_pct']}%")
            print(f"  → Missing values: {stats['missing_pct_df1']}% vs {stats['missing_pct_df2']}%")
            print(f"  → Key strength: {stats['potential_key_strength'].upper()}")
            print("─" * 50)

# ==================================== MAIN ========================================== #

def main():
    # Load all data sources
    branchwater_df = get_branchwater_results(branchwater_results_tsv)
    motupan_data = get_motupan_results(motupan_90_mmseqs_dir, motupan_90_wdir_dir)
    #genomad_data = get_genomad_results(genomad_raw_results_dir, genomad_parsed_results_dir)
    #eggnog_data = get_eggnog_results(eggnog_annotations_dir)
    pangenome_df = get_pangenome_results(pangenome_results_metadata_tsv)
    
    # Prepare DataFrames for linkage analysis
    analysis_dataframes = [
        branchwater_df,
        pangenome_df,
        #pd.DataFrame(genomad_data),  # Convert genomad dict to DataFrame
        pd.DataFrame(motupan_data[0]['m8_data']) if motupan_data else pd.DataFrame()  # Example motupan data
    ]
    
    dataframe_names = [
        'branchwater_metadata',
        'pangenome_metadata', 
        'genomad_annotations',
        'motupan_clusters'
    ]
    
    # Perform linkage analysis
    linkage_results = analyze_dataframe_linkage(
        analysis_dataframes, 
        dataframe_names,
        sample_size=500
    )
    
    # Print results
    print("\n" + "="*60)
    print("DATAFRAME LINKAGE ANALYSIS RESULTS")
    print("="*60)
    print_linkage_analysis(linkage_results)
    
    # Example integration using discovered links
    if 'branchwater_metadata <-> pangenome_metadata' in linkage_results:
        if 'accession' in linkage_results['branchwater_metadata <-> pangenome_metadata']:
            merged_df = pd.merge(
                branchwater_df,
                pangenome_df,
                on='accession',
                how='inner',
                suffixes=('_branchwater', '_pangenome')
            )
            print(f"\nSuccessfully merged datasets on 'accession'")
            print(f"Merged dataset shape: {merged_df.shape}")

if __name__ == "__main__":
    main()
  
