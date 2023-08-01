from os import listdir
from os.path import isfile, join
import os
import sys
sys.path.insert(0, './../')
from rdkit.Chem.rdMolTransforms import ComputeCentroid


from Models.ligdream1 import CompoundGenerator
import os
import torch



from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
import sys, os

def home():
    return os.getcwd()+'/ligdream/'

# write a function that takes a list of rdkit mol objects and saves as smi file into the input directory and names the file "generated_molecules.smi"
def save_molecules_to_smi(molecules, input_directory):
    # Create the file path
    file_path = input_directory.rstrip('/') + '/generated_molecules.smi'

    # Create a writer object to save molecules as SMI
    # writer = Chem.SmilesWriter(file_path)

    # Iterate over the list of molecules and write them to the SMI file
    with open(file_path, 'w') as f:
        for molecule in molecules:
            print(molecule)
            
            smiles = Chem.MolToSmiles(molecule)
            print(smiles)
            f.write("{}\n".format(smiles))
            

    print(file_path, " done writing")
    # Close the writer
    # writer.close()

thisdir = os.getcwd()

mypath=thisdir+"/PDBBind/"
folder=mypath
# print(mypath)


from os import walk


my_gen = CompoundGenerator(use_cuda=False)
vae_weights =  os.path.join(home(), "models/ligdream/modelweights/vae-210000.pkl")
encoder_weights =  os.path.join(home(), "models/ligdream/modelweights/encoder-210000.pkl")
decoder_weights =os.path.join(home(), "models/ligdream/modelweights/decoder-210000.pkl")

my_gen.load_weight(vae_weights, encoder_weights, decoder_weights)

subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
for i in subfolders[:100]:
    print(i)

ligands=[folder+"/"+folder[-4:]+"_ligand.mol2" for folder in subfolders]
proteins=[folder+"/"+folder[-4:]+"_protein.pdb" for folder in subfolders]
# filenames = next(walk(mypath), (None, None, []))[2]
print(len(subfolders))

for i,a in enumerate(subfolders):
    print(a)
    try:
        gen_dir=a+"/generated_compounds"
        os.mkdir(gen_dir)
    except:
        print("Already exists")

    seed_mol=Chem.MolFromMol2File(ligands[i])
    print(seed_mol)
    gen_mols = my_gen.generate_molecules(seed_mol,
                                     n_attemps=30,  # How many attemps of generations will be carried out
                                     lam_fact=1.,  # Variability factor
                                     probab=True,  # Probabilistic RNN decoding
                                     filter_unique_valid=True)  # Filter out invalids and replicates
    print(gen_mols)
    print ([Chem.MolToSmiles(a) for a in gen_mols])
    save_molecules_to_smi(gen_mols,gen_dir)