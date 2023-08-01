from os import listdir
from os.path import isfile, join
import os
import sys

import warnings
from pathlib import Path
import subprocess

import nglview as nv
from openbabel import pybel
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from rdkit import Chem

from opencadd.structure.core import Structure

sys.path.insert(0, './../')


def pdb_to_pdbqt(pdb_path, pdbqt_path, pH=7.4):
    """
    Convert a PDB file to a PDBQT file needed by docking programs of the AutoDock family.

    Parameters
    ----------
    pdb_path: str or pathlib.Path
        Path to input PDB file.
    pdbqt_path: str or pathlib.path
        Path to output PDBQT file.
    pH: float
        Protonation at given pH.
    """
    molecule = list(pybel.readfile("pdb", str(pdb_path)))[0]
    # add hydrogens at given pH
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)
    return

def smiles_to_pdbqt(smiles, pdbqt_path, pH=7.4):
    """
    Convert a SMILES string to a PDBQT file needed by docking programs of the AutoDock family.

    Parameters
    ----------
    smiles: str
        SMILES string.
    pdbqt_path: str or pathlib.path
        Path to output PDBQT file.
    pH: float
        Protonation at given pH.
    """
    molecule = pybel.readstring("smi", smiles)
    # add hydrogens at given pH
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    # generate 3D coordinates
    molecule.make3D(forcefield="mmff94s", steps=10000)
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)
    return

def sdf_to_pdbqt(smi_path, pdbqt_path, pH=7.4):
    """
    Convert a SMILES string to a PDBQT file needed by docking programs of the AutoDock family.

    Parameters
    ----------
    smiles: str
        SMILES string.
    pdbqt_path: str or pathlib.path
        Path to output PDBQT file.
    pH: float
        Protonation at given pH.
    """
    mol = pybel.readfile("sdf", smi_path)
    for molecule in mol:
        molecule=molecule
        break
#     molecule = pybel.readstring("sdf", smi_path)
    # add hydrogens at given pH
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    # generate 3D coordinates
#     molecule.make3D(forcefield="mmff94s", steps=10000)
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)
    return

def run_smina(
    ligand_path, protein_path, out_path, pocket_center, pocket_size,atom_terms, num_poses=10, exhaustiveness=8
):
    """
    Perform docking with Smina.

    Parameters
    ----------
    ligand_path: str or pathlib.Path
        Path to ligand PDBQT file that should be docked.
    protein_path: str or pathlib.Path
        Path to protein PDBQT file that should be docked to.
    out_path: str or pathlib.Path
        Path to which docking poses should be saved, SDF or PDB format.
    pocket_center: iterable of float or int
        Coordinates defining the center of the binding site.
    pocket_size: iterable of float or int
        Lengths of edges defining the binding site.
    num_poses: int
        Maximum number of poses to generate.
    exhaustiveness: int
        Accuracy of docking calculations.

    Returns
    -------
    output_text: str
        The output of the Smina calculation.
    """
    output_text = subprocess.check_output(
        [
            "smina",
            "--ligand",
            str(ligand_path),
            "--receptor",
            str(protein_path),
            "--out",
            str(out_path),
            "--center_x",
            str(pocket_center[0]),
            "--center_y",
            str(pocket_center[1]),
            "--center_z",
            str(pocket_center[2]),
            "--size_x",
            str(pocket_size[0]),
            "--size_y",
            str(pocket_size[1]),
            "--size_z",
            str(pocket_size[2]),
            "--num_modes",
            str(num_poses),
            "--exhaustiveness",
            str(exhaustiveness),
            "--atom_terms",
            atom_terms
        ],
        universal_newlines=True,  # needed to capture output text
    )
    return output_text

def minimise_smina(
    ligand_path, protein_path, out_path, atom_terms
):
    """
    Perform docking with Smina.

    Parameters
    ----------
    ligand_path: str or pathlib.Path
        Path to ligand PDBQT file that should be docked.
    protein_path: str or pathlib.Path
        Path to protein PDBQT file that should be docked to.
    out_path: str or pathlib.Path
        Path to which docking poses should be saved, SDF or PDB format.
    pocket_center: iterable of float or int
        Coordinates defining the center of the binding site.
    pocket_size: iterable of float or int
        Lengths of edges defining the binding site.
    num_poses: int
        Maximum number of poses to generate.
    exhaustiveness: int
        Accuracy of docking calculations.

    Returns
    -------
    output_text: str
        The output of the Smina calculation.
    """
    output_text = subprocess.check_output(
        [
            "smina",
            "--ligand",
            str(ligand_path),
            "--receptor",
            str(protein_path),
            "--out",
            str(out_path),
            "--score_only",
            "--atom_terms",
            str(atom_terms)
        ],
        universal_newlines=True,  # needed to capture output text
    )
    return output_text
thisdir = os.getcwd()

mypath=thisdir+"/PDBBind/"
folder=mypath
subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
ligands=[folder+"/"+folder[-4:]+"_ligand.mol2" for folder in subfolders]
proteins=[folder+"/"+folder[-4:]+"_protein.pdb" for folder in subfolders]

for i,a in enumerate(subfolders[:100]):
    print(i, 'start preparation', a)
    lig=Chem.rdmolfiles.MolFromMol2File(ligands[i], sanitize=True)
    centroid = ComputeCentroid(lig.GetConformer())
    centroid=[centroid.x, centroid.y, centroid.z]
    mol=a+'/mol.pdbqt'
    prot=proteins[i]+"qt"
    # print(ligands[i][:-4]+'sdf')
    try:
        sdf_to_pdbqt(ligands[i][:-4]+'sdf',mol)
        gen_mol_dir=a+'/generated_compounds_SQUID/'
        box=[25,25,25]
        print("start minimising")
        print(proteins[i])
        print(prot[i])
        pdb_to_pdbqt(proteins[i],prot)
        minimise_smina(mol, prot, a+"/dock_pose_crystal.sdf",a+"/mol_int.pdb")
        print("start protein to pdbqt")
        
        f = open(gen_mol_dir+'generated_molecules.smi', "r")
        suppl=f.read().split()
        for b,smile in enumerate(suppl):
            smi_path=gen_mol_dir+str(b)+'_mol.pdbqt'
            # print(smile, smi_path)
            print("start smile to pdbqt")
            smiles_to_pdbqt(smile,smi_path)
            out_path=gen_mol_dir+str(b)+'_docked.sdf'
            print("start dock")
            run_smina(smi_path,prot,out_path,centroid,box,gen_mol_dir+str(b)+'_docked_mol_inter.pdb')
            print("finish dock")
    except:
        print("Fail", a )
   
 
        