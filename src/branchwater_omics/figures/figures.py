# ===================================== IMPORTS ====================================== #

import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==================================== FUNCTIONS ===================================== #

def save_figure(output_path):
    """
    Save the current figure to the specified output path, creating directories 
    if needed.
    """
    directory = os.path.dirname(output_path)
    if directory:  # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
