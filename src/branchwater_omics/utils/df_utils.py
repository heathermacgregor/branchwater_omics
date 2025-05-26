# ===================================== IMPORTS ====================================== #

from collections import Counter
import re
from typing import List, Tuple, Set

import numpy as np
import pandas as pd

# ==================================== FUNCTIONS ===================================== #

def find_common_substrings(
    df: pd.DataFrame,
    column: str,
    n_gram_range: Tuple[int, int] = (1, 3),
    top_n: int = 20,
    stopwords: Set[str] = None,
    min_term_length: int = 3,
    remove_punctuation: bool = True
) -> pd.DataFrame:
    """
    Identify the most common substrings in a text column using n-gram analysis.
    
    Args:
        df: Input DataFrame
        column: Name of text column to analyze
        n_gram_range: Range of n-gram sizes to consider (min, max)
        top_n: Number of top results to return
        stopwords: Words/phrases to exclude from counting
        min_term_length: Minimum length of terms to include
        remove_punctuation: Whether to remove punctuation
    
    Returns:
        Top substrings with counts, sorted by frequency
    """
    # Default stopwords if none provided
    if stopwords is None:
        stopwords = {'', 'na', 'none', 'unknown', 'not', 'collected', 'sample'}
    
    # Clean and preprocess text
    clean_series = (
        df[column]
        .dropna()
        .astype(str)
        .str.lower()
    )
    
    if remove_punctuation:
        clean_series = clean_series.str.replace(r'[^\w\s]', '', regex=True)
    
    # Generate n-grams
    all_ngrams = []
    for text in clean_series:
        words = text.split()
        for n in range(n_gram_range[0], n_gram_range[1] + 1):
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            filtered = [
                ng for ng in ngrams 
                if (len(ng) >= min_term_length) and (ng not in stopwords)
            ]
            all_ngrams.extend(filtered)
    
    # Count and sort
    counts = Counter(all_ngrams)
    result_df = pd.DataFrame(
        counts.most_common(top_n), 
        columns=['substring', 'count']
    )
    
    return result_df


def parse_coordinate(coord_str):
    """
    Convert strings like "46.17467 N 123.8493 W" to signed decimal degrees.
    Returns (lat, lon) or (NaN, NaN) if invalid.
    """
    if pd.isna(coord_str) or coord_str.lower() == "none":
        return (np.nan, np.nan)
    
    # Extract numbers and directions using regex
    matches = re.findall(r"([\d.]+)\s*([NS]?)[,\s]*([\d.]+)\s*([EW]?)", coord_str, re.IGNORECASE)
    
    if not matches:
        return (np.nan, np.nan)
    
    lat_val, lat_dir, lon_val, lon_dir = matches[0]
    
    try:
        lat = float(lat_val) * (-1 if lat_dir.upper() in ('S', 'W') else 1)
        lon = float(lon_val) * (-1 if lon_dir.upper() in ('W', 'S') else 1)
        return (lat, lon)
    except:
        return (np.nan, np.nan)
