# SG-ATT
This is a deep learning model designed for predicting molecular properties. Initially, the model takes into account the dimensional features of molecules, allowing it to capture the structure and composition of the molecules. Furthermore, by incorporating molecular knowledge graphs, the model leverages the relationships between atoms within the molecules, thereby enhancing its predictive capabilities for molecular properties.

#Environment:  
python=3.6.8  
pytorch=1.0.1  
cudatoolkit=9.0  
rdkit=2018.09.1.0  
scikit-learn=0.20.2  
torch=1.7.1  
dgl=0.6.1  
dgllife=0.2.8  
pandarallel=1.5.2  
numpy=1.20.3  
pandas=1.3.1  
lmdb=1.2.1  
tqdm  
subword-nmt  

#Run SG-ATT:  
python train.py --SGATT
