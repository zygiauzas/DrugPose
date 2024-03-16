# DrugPose: Benchmarking 3D generative methods for early stage drug discovery

![alt text](Images/Pipeline.png)

## Data availability

Relevant data can be downloaded from PDBBind: http://www.pdbbind.org.cn/ and it should be uploaded to PDBBind folder.

## Models

Each of the model should be downloaded and the individual environments should be set up separately:

- Ligdream: \url{https://github.com/playmolecule/ligdream}
- SQUID: \url{https://github.com/keiradams/SQUID}
- Pocket2mol: \url{https://github.com/pengxingang/Pocket2Mol}

## Processing

### Setting up the directories
Create a directory structure as outlined in the project documentation:

|–– DrugPose

|–– FLAG

|–– Pocket2Mol

|–– SQUID

|–– ligdream

|–– PDB_Bind

|  |…

|  |–– 1ppi

|  |  |–– generated_compounds

|  |  |–– generated_compounds_SQUID

|  |…

### Running the models on the PDBbind
Once the repos are set up you can run the relevant scripts for Ligdream, SQUID and Pocket2mol models.

### Running the evaluation script
Once the molecules are generated you can run the analysis scripts.

