import numpy as np

def load_pdb_coordinates_from_file(file):
    """
    Loads atomic coordinates from an uploaded PDB file object (not filename).
    """
    coords = []
    content = file.read().decode('utf-8').splitlines()
    for line in content:
        if line.startswith(('ATOM', 'HETATM')):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
    return np.array(coords)

def load_pdb_atom_and_hetatm_coordinates_from_file(file):
    """
    Loads ATOM and HETATM coordinates separately from a PDB file object.
    Returns two NumPy arrays: atom_coords and hetatm_coords.
    """
    atom_coords = []
    hetatm_coords = []
    content = file.read().decode('utf-8').splitlines()
    for line in content:
        if line.startswith('ATOM'):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atom_coords.append([x, y, z])
        elif line.startswith('HETATM'):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            hetatm_coords.append([x, y, z])
    return np.array(atom_coords), np.array(hetatm_coords)
