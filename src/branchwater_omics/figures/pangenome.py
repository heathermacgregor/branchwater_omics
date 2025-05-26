# ===================================== IMPORTS ====================================== #

import seaborn as sns
import matplotlib.pyplot as plt
import os

# ================================== LOCAL IMPORTS =================================== #

from figures import save_figure

# ==================================== FUNCTIONS ===================================== #

# Visualize CheckM Quality Metrics
def completeness_vs_contamination(df, output_path):
    """
    Create a scatter plot of completeness vs. contamination to identify clusters 
    of high-quality genomes.
    """
    sns.scatterplot(
        data=df,
        x='checkm_completeness',
        y='checkm_contamination',
        hue='gtdb_type_species_of_genus',  # Highlight type species
        alpha=0.6
    )
    plt.axvline(90, linestyle='--', color='gray')  # MIMAG completeness cutoff
    plt.axhline(5, linestyle='--', color='gray')   # MIMAG contamination cutoff
    save_figure(output_path)

def contig_vs_scaffold(df, valid_levels, output_path):
    sns.boxplot(
        data=df,
        x='ncbi_assembly_level',
        y='n50_contigs',
        order=valid_levels  # Ensure order matches filtered data
    )
    plt.yscale('log')
    save_figure(output_path)

def map(df, output_path):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['lon'], df['lat'], alpha=0.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Genome Sampling Locations')
    save_figure(output_path)
