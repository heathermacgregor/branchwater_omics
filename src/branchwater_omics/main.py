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
    combine_loaded_data,
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
    
    subset_file_groups = dict(list(file_groups.items())[:2])
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
        df_clean, 
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

def analyze_dataframe_linkage(dataframes, dataframe_names, sample_size=1000, 
                             cross_threshold=50, min_unique=10):
    """
    Analyze potential linkages between DataFrames through:
    - Direct column matches (shared column names)
    - Cross-column value matches (different names, similar values)
    - Data type compatibility
    - Value overlap analysis
    
    Args:
        dataframes (list): List of pandas DataFrames
        dataframe_names (list): Names for each DataFrame
        sample_size (int): Number of values to sample per column
        cross_threshold (int): Minimum overlap percentage to consider cross-column matches
        min_unique (int): Minimum unique values required for column comparison
    
    Returns:
        dict: Detailed linkage analysis with match quality metrics
    """
    analysis_results = {}
    
    # Precompute sampled values and unique counts for all columns
    column_profiles = []
    for df in dataframes:
        profile = {}
        for col in df.columns:
            # Clean and sample column values
            clean_series = df[col].dropna().astype(str).str.strip().str.lower()
            unique_count = clean_series.nunique()
            
            profile[col] = {
                'sample': set(clean_series.sample(min(sample_size, len(clean_series)))) 
                           if not clean_series.empty else set(),
                'dtype': df[col].dtype,
                'unique': unique_count,
                'missing_pct': df[col].isna().mean() * 100
            }
        column_profiles.append(profile)

    # Compare all DataFrame pairs
    for i, (df1, name1) in enumerate(zip(dataframes, dataframe_names)):
        for j, (df2, name2) in enumerate(zip(dataframes, dataframe_names)):
            if i >= j:  # Avoid duplicate comparisons
                continue
                
            pair_key = f"{name1} <-> {name2}"
            pair_result = {'direct_matches': {}, 'cross_matches': []}
            profile1 = column_profiles[i]
            profile2 = column_profiles[j]

            # 1. Check direct column name matches
            common_cols = set(df1.columns) & set(df2.columns)
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

            # 2. Find cross-column value matches
            for col1 in df1.columns:
                for col2 in df2.columns:
                    if col1 == col2 or col1 in common_cols or col2 in common_cols:
                        continue  # Skip direct matches
                        
                    # Get column profiles
                    p1 = profile1.get(col1, {})
                    p2 = profile2.get(col2, {})
                    
                    # Basic validity checks
                    if not p1 or not p2 or p1['unique'] < min_unique or p2['unique'] < min_unique:
                        continue
                    
                    # Calculate value overlap
                    overlap = len(p1['sample'] & p2['sample'])
                    if overlap == 0:
                        continue
                        
                    # Calculate overlap percentages
                    max_sample = max(len(p1['sample']), len(p2['sample']))
                    min_sample = min(len(p1['sample']), len(p2['sample']))
                    overlap_pct_min = (overlap / min_sample) * 100
                    overlap_pct_max = (overlap / max_sample) * 100
                    
                    if overlap_pct_min >= cross_threshold:
                        # Calculate full dataset statistics
                        exact_matches = (
                            df1[col1].astype(str).str.lower().isin(
                                df2[col2].astype(str).str.lower()
                            ).mean() * 100
                        )
                        
                        # Type compatibility check
                        type_compatible = pd.api.types.is_dtype_equal(p1['dtype'], p2['dtype'])
                        
                        match_strength = 'strong' if (overlap_pct_min > 75 and type_compatible) \
                            else 'moderate' if (overlap_pct_min > 50) else 'weak'
                        
                        pair_result['cross_matches'].append({
                            'columns': (col1, col2),
                            'overlap_pct_min': round(overlap_pct_min, 1),
                            'overlap_pct_max': round(overlap_pct_max, 1),
                            'exact_match_pct': round(exact_matches, 1),
                            'dtypes': (str(p1['dtype']), str(p2['dtype'])),
                            'type_compatible': type_compatible,
                            'strength': match_strength,
                            'missing_pct': (round(p1['missing_pct'], 1), 
                                          round(p2['missing_pct'], 1)),
                            'unique_values': (p1['unique'], p2['unique'])
                        })

            analysis_results[pair_key] = pair_result
            
    return analysis_results

def print_linkage_analysis(results):
    """Enhanced printing with cross-column matches"""
    for pair, data in results.items():
        print(f"\n=== LINKAGE ANALYSIS: {pair} ===")
        
        # Direct matches
        if data['direct_matches']:
            print("\nDIRECT COLUMN MATCHES:")
            for col, stats in data['direct_matches'].items():
                print(f"  Column: {col}")
                print(f"  → Type: {stats['dtypes'][0]} vs {stats['dtypes'][1]}")
                print(f"  → Overlap: {stats['overlap_pct']}%")
                print(f"  → Strength: {stats['strength'].upper()}")
                print("-" * 50)
                
        # Cross matches
        if data['cross_matches']:
            print("\nPOTENTIAL CROSS-COLUMN MATCHES:")
            for match in sorted(data['cross_matches'], 
                              key=lambda x: (-x['overlap_pct_min'], x['columns'])):
                print(f"  {match['columns'][0]} ({match['dtypes'][0]})  ↔  "
                      f"{match['columns'][1]} ({match['dtypes'][1]})")
                print(f"  → Overlap (min/max): {match['overlap_pct_min']}%/{match['overlap_pct_max']}%")
                print(f"  → Full dataset match: {match['exact_match_pct']}%")
                print(f"  → Type compatible: {'Yes' if match['type_compatible'] else 'No'}")
                print(f"  → Missing values: {match['missing_pct'][0]}% vs {match['missing_pct'][1]}%")
                print(f"  → Unique values: {match['unique_values'][0]} vs {match['unique_values'][1]}")
                print(f"  → Strength: {match['strength'].upper()}")
                print("-" * 50)

def save_linkage_analysis(results, output_path):
    """Save linkage analysis results to specified file path"""
    with open(output_path, 'w') as f:
        # Write header
        f.write("="*80 + "\n")
        f.write("BRANCHWATER OMICS DATASET LINKAGE REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Write analysis content
        for pair, data in results.items():
            f.write(f"\n=== LINKAGE ANALYSIS: {pair} ===\n")
            
            # Direct matches section
            if data['direct_matches']:
                f.write("\nDIRECT COLUMN MATCHES:\n")
                for col, stats in data['direct_matches'].items():
                    f.write(f"  Column: {col}\n")
                    f.write(f"  → Data types: {stats['dtypes'][0]} vs {stats['dtypes'][1]}\n")
                    f.write(f"  → Sample overlap: {stats['overlap_pct']}%\n")
                    f.write(f"  → Key strength: {stats['strength'].upper()}\n")
                    f.write("-"*60 + "\n")
            
            # Cross matches section
            if data['cross_matches']:
                f.write("\nPOTENTIAL CROSS-COLUMN MATCHES:\n")
                for match in sorted(data['cross_matches'], key=lambda x: (-x['overlap_pct_min'], x['columns'])):
                    col1, col2 = match['columns']
                    f.write(f"  {col1} ({match['dtypes'][0]})  ↔  {col2} ({match['dtypes'][1]})\n")
                    f.write(f"  → Overlap (min/max): {match['overlap_pct_min']}%/{match['overlap_pct_max']}%\n")
                    f.write(f"  → Full dataset match: {match['exact_match_pct']}%\n")
                    f.write(f"  → Type compatible: {'Yes' if match['type_compatible'] else 'No'}\n")
                    f.write(f"  → Unique values: {match['unique_values'][0]} vs {match['unique_values'][1]}\n")
                    f.write(f"  → Strength: {match['strength'].upper()}\n")
                    f.write("-"*60 + "\n")
                    
            f.write("\n" + "="*80 + "\n")
            
# ==================================== MAIN ========================================== #

def main():
    # Load all data sources
    branchwater_df = get_branchwater_results(branchwater_results_tsv)
    motupan_data = get_motupan_results(motupan_90_mmseqs_dir, motupan_90_wdir_dir)
    motupan_df = combine_loaded_data(motupan_data)
    print(motupan_df.head())
    """
    #genomad_data = get_genomad_results(genomad_raw_results_dir, genomad_parsed_results_dir)
    #eggnog_data = get_eggnog_results(eggnog_annotations_dir)
    pangenome_df = get_pangenome_results(pangenome_results_metadata_tsv)
    
    # Prepare DataFrames for linkage analysis
    analysis_dataframes = [
        #branchwater_df,
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
    
    # Print to console
    print("\n" + "="*60)
    print("DATAFRAME LINKAGE ANALYSIS RESULTS")
    print("="*60)
    print_linkage_analysis(linkage_results)
    
    # Save to file
    output_path = '/global/homes/m/macgrego/branchwater_omics/merge_columns.txt'
    save_linkage_analysis(linkage_results, output_path)
    print(f"\nFull analysis saved to: {output_path}")
    
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
    """

if __name__ == "__main__":
    main()
  
