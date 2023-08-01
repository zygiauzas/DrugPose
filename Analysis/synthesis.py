import numpy as np
import matplotlib.pyplot as plt

import os
from rdkit import Chem 
import pandas as pd
import numpy as np
from smallworld_api import SmallWorld

from rdkit import Chem
from rdkit.Chem import Crippen


thisdir = os.getcwd()

mypath=thisdir+"/PDBBind/"
folder=mypath
subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
ligands=[folder+"/"+folder[-4:]+"_ligand.mol2" for folder in subfolders]
sw = SmallWorld()
total=[]
total_s=[]
Ro5=[]


for a in subfolders:
    generated_mol_loc=a+'/generated_compounds/generated_molecules.smi'
    try:
        f = open(generated_mol_loc, "r")
    except:
        print("dir not found")
        continue
    suppl=f.read().split()
    print("dir found")
    for b,smile in enumerate(suppl):
                    
        try:
            results : pd.DataFrame = sw.search(smile, dist=1, db=sw.REAL_dataset)

            total_s.append(results.iloc[0]['dist'])
        except:
            total_s.append(0)

    total.append(np.array(total_s))
total=np.array(total, dtype=object)
print(total)

np.save('s_ligdream_check.npy', total)

squid=np.load('s_squid_check.npy',allow_pickle=True)[-1]
p2m=np.load('s_p2m_check.npy',allow_pickle=True)[-1]
ligdream=np.load('s_ligdream_check.npy',allow_pickle=True)[-1]
print(squid)
print("amount of zero for squid:",np.sum(squid==0)/len(squid))
print("amount of zero for ligdream:",np.sum(ligdream==0)/len(ligdream))
print("amount of zero for p2m:",np.sum(p2m==0)/len(p2m))
squid=squid[squid!=0]
p2m=p2m[p2m!=0]
ligdream=ligdream[ligdream!=0]



# Create a histogram
plt.hist(squid,label='SQUID',alpha=0.6,weights=np.ones(len(squid)) / len(squid))
plt.hist(p2m,label='Pocket2mol',alpha=0.6,weights=np.ones(len(p2m)) / len(p2m))
plt.hist(ligdream,label='Ligdream',alpha=0.6,weights=np.ones(len(ligdream)) / len(ligdream))
print(np.sum(squid==1)/len(squid))
print(np.sum(ligdream==1)/len(ligdream))
print(np.sum(p2m==1)/len(p2m))
plt.legend()
# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')

# Display the histogram
plt.show()