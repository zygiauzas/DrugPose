import os
from rdkit import Chem 
import pandas as pd
import numpy as np

def from_pdb_to_df(path):
    """
    Parses a PDB file and converts atom data into pandas DataFrames.

    Parameters:
        path (str): The file path to the PDB file.

    Returns:
        list: A list of pandas DataFrames, each containing atom information from a different section of the PDB file.
    """

    # Open the PDB file and read its content
    with open(path, "r") as f:
        file = f.read().split('END')

    # Split the file content into sections (models or chains)
    files = [a.split('\n') for a in file]

    # Initialize the list to store DataFrames for each section
    list_of_df = []

    # Loop through each section
    for a in files[:-1]:
        # Initialize the list to store atom information for the current section
        atom_terms = []

        # Loop through each line in the current section
        for line in a[1:]:
            # Break the loop if an empty line is encountered, indicating the end of the current section
            if line == '':
                break

            # Check if the line contains atom data (starts with a space)
            if line[0] == ' ':
                # Parse the fields in the line
                fields = line.strip().split()
                atom_id = int(fields[0])
                atom_type = fields[1]
                pos = fields[2][1:-1].split(',')
                pos1, pos2, pos3 = [float(a) for a in pos]
                explain = [float(fields[i]) for i in range(3, 8)]

                # Append atom information as a tuple to the current section's list
                atom_terms.append((atom_id, atom_type, pos1, pos2, pos3, explain[0], explain[1], explain[2], explain[3], explain[4]))

        # Create a DataFrame for the current section's atom information
        tempo = pd.DataFrame(atom_terms, columns=['id', 'atomtype', 'pos_x', 'pos_y', 'pos_z', 'int_gauss', 'int_gauss',
                                                  'int_repul', 'int_hydrophobic', 'non_dir_h_bond'])

        # Append the DataFrame to the list of DataFrames
        list_of_df.append(tempo.copy())

    # Return the list of DataFrames containing atom information for each section
    return list_of_df


def compare_interactions(mol, list_mol):
    """
    Compares the interactions between a molecule and a list of molecules by calculating the distances
    between atoms in each pair of molecules.

    Parameters:
        mol (list of pandas DataFrame): The molecule as a list of pandas DataFrames, each containing atom information.
        list_mol (list of pandas DataFrame): A list of molecules as lists of pandas DataFrames, each containing atom information.

    Returns:
        list: A list of 3D arrays, where each array represents the distances between atoms in the 'mol' molecule and
              each molecule in 'list_mol'.
    """

    # Initialize a list to store the distances between atoms
    list_of_dist = []

    # Loop through each molecule in 'list_mol'
    for a in list_mol:
        # Extract the 3D coordinates of atoms from 'mol' and the current molecule in 'list_mol'
        molnp1 = mol[0].iloc[:, 2:5].to_numpy()
        molnp2 = a.iloc[:, 2:5].to_numpy()

        # Initialize a list to store the distances for each atom in 'mol' with respect to the current molecule in 'list_mol'
        temp_dist = []

        # Loop through each atom in 'mol'
        for b in molnp1:
            # Repeat the current atom's coordinates to match the shape of 'molnp2'
            b1 = np.tile(b.reshape(-1, 3), (molnp2.shape[0], 1))

            # Calculate the distances between the current atom in 'mol' and all atoms in the current molecule in 'list_mol'
            dist = np.linalg.norm(molnp2 - b1, axis=1)

            # Append the distances to the temporary list
            temp_dist.append(dist)

        # Convert the temporary list of distances into a 3D numpy array and append it to 'list_of_dist'
        temp = np.array(temp_dist)
        list_of_dist.append(temp)

    return list_of_dist


def from_list_of_dist_to_max_indexes(list_of_dist):
    """
    Converts a list of distance arrays to the indexes of maximum values within each array.

    Parameters:
        list_of_dist (list of 3D numpy arrays): A list of 3D numpy arrays, where each array contains distances between atoms.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - max_ind_list: A numpy array with shape (n, 2) where 'n' is the number of distance arrays in 'list_of_dist'.
                            It contains indexes of the maximum values within each distance array. If the maximum value
                            is greater than the threshold, the second element in the sub-array is set to NaN.
            - max_ind: A 1D numpy array with shape (n,) containing the index of the minimum value for each distance array.
    """

    # Find the indexes of the maximum values in each distance array
    max_ind = np.argmin(list_of_dist, axis=1)
    max_ind1 = np.argmin(list_of_dist.T, axis=1)

    # Initialize lists to store the maximum indexes and maximum indexes with threshold check
    max_ind_list = []
    max_ind_list1 = []

    # Set a threshold for the distance values
    threshold = 2

    # Iterate through each distance array
    for i, a in enumerate(list_of_dist):
        # Check if the distance value at the maximum index is greater than the threshold
        if a[max_ind[i]] > threshold:
            max_ind_list.append([i, np.nan])  # If greater, set the second element to NaN
        else:
            max_ind_list.append([i, max_ind[i]])  # If within threshold, append the maximum index

    # Convert the lists to numpy arrays
    max_ind_list = np.array(max_ind_list)

    # Return the tuple containing the maximum indexes with threshold check and the raw maximum indexes
    return max_ind_list, max_ind


def from_ind_to_interaction_comparison(max_ind_list, temp, temp1, i):
    d=1
    """
    Compares interactions between two molecules based on their maximum index lists.

    Parameters:
        max_ind_list (numpy array): A numpy array containing pairs of indexes indicating the maximum values within each distance array.
        temp (list of pandas DataFrame): A list of pandas DataFrames, each containing atom information for a molecule.
        temp1 (list of pandas DataFrame): Another list of pandas DataFrames, each containing atom information for a different molecule.
        i (int): Index of the molecule in 'temp' for which interactions are being compared with 'temp1'.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - diff: A numpy array with the absolute differences between the interaction properties of the two molecules.
            - diff_bool: A boolean numpy array indicating whether the absolute differences are within a threshold (1).
    """

    # Filter out pairs with NaN values in the second element of the sub-array
    x = max_ind_list[~pd.isnull(max_ind_list[:, 1])]

    # Extract interaction properties of the molecules at the specified indexes
    item1 = temp1[0].iloc[x[:, 0]].iloc[:, -5:].astype(float).to_numpy()
    item2 = temp[i].iloc[x[:, 1]].iloc[:, -5:].astype(float).to_numpy()

    # Calculate the differences between the interaction properties of the two molecules
    diff = item1 - item2

    # Check if the differences are within a threshold (1)
    diff_bool = item2*d<diff

    return diff, diff_bool

# The following line saves the current working directory to the variable 'thisdir'
thisdir = os.getcwd()


mypath=thisdir+"/PDBbind/"
folder=mypath
subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
ligands=[folder+"/"+folder[-4:]+"_ligand.mol2" for folder in subfolders]
proteins=[folder+"/"+folder[-4:]+"_protein.pdb" for folder in subfolders]
result_list=[]
result_list_per_molecule=[]
for i,a in enumerate(subfolders[:100]):
    # print(i, 'start preparation', a)
    lig=Chem.rdmolfiles.MolFromMol2File(ligands[i], sanitize=True)
    seed_mol=a+'/mol_int.pdb'
    try:
        files = [ f.path for f in os.scandir(a+'/generated_compounds_SQUID/')]
    except:
        print("stop")
        continue
    pdb_files=[f for f in files if f[-3:]=="pdb"]
    score_per_folder=[]
    if pdb_files==[]:
        continue
    for file in pdb_files:
        print(file)
        temp1=from_pdb_to_df(seed_mol)
        temp=from_pdb_to_df(file)
        list_of_dist=compare_interactions(temp1, temp)
        scores=[]
        for i,dist in enumerate(list_of_dist):
            max_ind_list, max_ind=from_list_of_dist_to_max_indexes(dist)
            diff=from_ind_to_interaction_comparison(max_ind_list,temp,temp1,i)
            scores.append(np.all(diff_bool,axis=1).sum()/len(temp1[0])*100)
            print("temp result",np.all(diff_bool,axis=1).sum()/len(temp1[0])*100)
        score_per_folder.append(np.array(scores))
    try:
        result=np.array(score_per_folder, dtype=object)
        something=result>40
        print(a[-4:],"results: ",np.any(something,axis=1).sum()/len(pdb_files)*100)
        result_list.append(np.any(something,axis=1).sum()/len(pdb_files)*100)
        
    except:
        print(a,"failed")
result=(np.array(result_list)==0).sum()/len(result_list)*100
result1=(np.array(result_list).mean())
print(result, "for this many pdbs failed")
print(result1, "success rate")
    