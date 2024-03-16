# DrugPose: Benchmarking 3D generative methods for early stage drug discovery

![alt text](Images/Pipeline.png)

## Data availability

Relevant data can be downloaded from PDBBind: http://www.pdbbind.org.cn/ and it should be uploaded to PDBBind folder.

## Models

Each of the model should be downloaded and the individual environments should be set up separately:

\item Ligdream: \url{https://github.com/playmolecule/ligdream}
\item SQUID: \url{https://github.com/keiradams/SQUID}
\item Pocket2mol: \url{https://github.com/pengxingang/Pocket2Mol}

## Processing

### Setting up the directories
Create a directory structure as outlined in the project documentation:
$ ./tree-md .

.
 * [tree-md](./tree-md)
 * [dir2](./dir2)
   * [file21.ext](./dir2/file21.ext)
   * [file22.ext](./dir2/file22.ext)
   * [file23.ext](./dir2/file23.ext)
 * [dir1](./dir1)
   * [file11.ext](./dir1/file11.ext)
   * [file12.ext](./dir1/file12.ext)
 * [file_in_root.ext](./file_in_root.ext)
 * [README.md](./README.md)
 * [dir3](./dir3)

### Running the models on the PDBbind
Once the repos are set up you can run the relevant scripts for Ligdream, SQUID and Pocket2mol models.
### Running the evaluation script
Once the molecules are generated you can run the analysis scripts.

