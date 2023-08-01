from os import listdir
from os.path import isfile, join
import os
import sys
sys.path.insert(0, './../')
from rdkit.Chem.rdMolTransforms import ComputeCentroid


from ligdream1 import CompoundGenerator
import os
import torch



from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
import sys, os

def home():
    return os.getcwd()

thisdir = os.getcwd()

mypath=thisdir+"/PDBBind/"
folder=mypath
subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
ligands=[folder+"/"+folder[-4:]+"_ligand.mol2" for folder in subfolders]
proteins=[folder+"/"+folder[-4:]+"_protein.pdb" for folder in subfolders]

for i,a in enumerate(subfolders):
    
    seed_mol=Chem.MolFromMol2File(ligands[i])
    centroid = ComputeCentroid(seed_mol.GetConformer())
    # centroid=[centroid.x, centroid.y, centroid.z]
    # print(i, 'python sample_for_pdb.py --pdb_path', proteins[i], '--center ' ,centroid.x,',', centroid.y,',', centroid.z)
    os.system(str(i, 'python sample_for_pdb.py --pdb_path', proteins[i], '--center ' ,centroid.x,',', centroid.y,',', centroid.z))
    # print(i,a,centroid)