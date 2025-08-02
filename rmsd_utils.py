import numpy as np

def calculate_rmsd_matrix(all_data):
    """
    Calculate RMSD of each structure in all_data with respect to the first structure.
    all_data: a list or NumPy array of shape (n_frames, n_atoms * 3)
    Returns: list of RMSD values
    """
    reference = all_data[0]  # Use first structure as reference
    rmsd_values = []

    for structure in all_data:
        diff = structure - reference
        rmsd = np.sqrt(np.mean(np.square(diff)))
        rmsd_values.append(rmsd)

    return rmsd_values
