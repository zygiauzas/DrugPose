from rdkit import Chem
from rdkit.Chem import Crippen
import numpy as np

def check_ghose_filter(smiles):
    # Calculate the number of hydrogen bond donors
    molecule = Chem.MolFromSmiles(smiles)
    num_donors = Chem.rdMolDescriptors.CalcNumHBD(molecule)
    
    # Calculate the number of hydrogen bond acceptors
    num_acceptors = Chem.rdMolDescriptors.CalcNumHBA(molecule)
    refractivity = Crippen.MolMR(molecule)
    num_atoms = molecule.GetNumAtoms()
    
    # Calculate the molecular weight
    molecular_weight = Chem.rdMolDescriptors.CalcExactMolWt(molecule)
    
    # Calculate the logP value
    logp_value = Crippen.MolLogP(molecule)
    
    # Ghose filter criteria
    # Partition coefficient log P in -0.4 to 5.6 range
    # Molar refractivity from 40 to 130
    # Molecular weight from 160 to 480
    # Number of atoms from 20 to 70 (includes H-bond donors [e.g. OHs and NHs] and H-bond acceptors [e.g. Ns and Os])
    ghose_violations = []
    if logp_value < -0.4 or logp_value > 5.6:
        ghose_violations.append("Partition coefficient log P is outside the range of -0.4 to 5.6")
    if refractivity < 40 or refractivity > 130:
        ghose_violations.append("Molar refractivity is outside the range of 40 to 130")
    if molecular_weight < 160 or molecular_weight > 480:
        ghose_violations.append("Molecular weight is outside the range of 160 to 480")
    if num_atoms < 20 or num_atoms > 70:
        ghose_violations.append("Number of atoms is outside the range of 20 to 70")
    
    return ghose_violations

thisdir = os.getcwd()

mypath=thisdir+"/PDBBind/"
folder=mypath
subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
ligands=[folder+"/"+folder[-4:]+"_ligand.mol2" for folder in subfolders]

total=[]
total_s=[]
Ro5=[]
mol_l=[]
outputs=thisdir+'/Pocket2mol/outputs/'
subsubfolders = [ f.path for f in os.scandir(outputs) if f.is_dir() ]
for a in subsubfolders:
    logtxt=a+'/log.txt'
    f = open(logtxt, "r")
    log=f.read().splitlines()
    # print(log)
    for i,a in enumerate(log):
        a=a.split()
        if a[-2] =='Success:':
            mol_l.append(a[-1])
            a[-1]

            if check_ghose_filter(a[-1])==[]:
                total_s.append(1)
            else:
                total_s.append(0)

total=np.array(total_s)
print(total)
print(np.sum(np.array(total))/len(total))