import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from torch.utils import data
from encoder.graph_features import atom_features
from collections import defaultdict
from subword_nmt.apply_bpe import BPE
import codecs
from rdkit import RDLogger
import logging
logger = logging.getLogger()
RDLogger.DisableLog('rdApp.*')
import dgl
import torch
import pdb
import pickle
from .load_triples import Triples
from .load_fratriples import Fra_Triples


class MolGraphDataset(data.Dataset):
    def __init__(self, path, prediction=False):
        print(path)
        file=pd.read_csv(path,sep=',')
        n_cols=file.shape[1]
        self.header_cols = np.genfromtxt(path, delimiter=',', usecols=range(0, n_cols), dtype=np.str, comments=None)
        self.target_names = self.header_cols[0:1, -1]
        self.smiles1 = np.genfromtxt(path,delimiter=',',skip_header=1,usecols=[0],dtype=np.str,comments=None)

        if prediction:
            self.targets = np.empty((len(self.smiles1),1))
        else:
            self.targets = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=[1], dtype=np.int, comments=None)


    def __getitem__(self, index):

        adj_1, nd_1, ed_1 = smile_to_graph(self.smiles1[index])
        d1, mask_1 = drug2emb_encoder(self.smiles1[index])

        targets = self.targets[index]

        bg = smiles_2_kgdgl(self.smiles1[index])

        loaded_dict = pickle.load(open('/home/ntu/PycharmProjects/T-KG/SAGTT/encoder/RotatE_128_64_emb.pkl', 'rb'))
        entity_emb, relation_emb = loaded_dict['entity_emb'],loaded_dict['relation_emb']

        return (adj_1, nd_1, ed_1), targets, d1, mask_1,bg,entity_emb, relation_emb

    def __len__(self):
        return len(self.smiles1)

def bondtype_features(bond):
    bondtype_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    bond2emb = {}
    for idx, bt in enumerate(bondtype_list):
        bond2emb[bt] = idx
    fbond = bond2emb[str(bond.GetBondType())]
    return fbond

def smiles_2_kgdgl(smiles):  # augmented graph
    data1 = Triples()
    data2 = Fra_Triples()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        print('Invalid mol found')
        return None

    connected_atom_list = []
    for bond in mol.GetBonds():
        connected_atom_list.append(bond.GetBeginAtomIdx())
        connected_atom_list.append(bond.GetEndAtomIdx())

    connected_atom_list = sorted(list(set(connected_atom_list)))
    connected_atom_map = {k: v for k, v in zip(connected_atom_list, list(range(len(connected_atom_list))))}
    atoms_feature = [0 for _ in range(len(connected_atom_list))]

    # get all node ids and relations
    begin_attributes = []  # attributes
    end_atoms = []  # atoms
    rel_features = []  # relations between attributes and atoms
    fileHandler = open("/home/ntu/PycharmProjects/T-KG/SAGTT/encoder/Fragment_triples.txt", "r")
    lines = fileHandler.readlines()
    for atom in mol.GetAtoms():
        node_index = atom.GetIdx()
        symbol = atom.GetSymbol()
        atomicnum = atom.GetAtomicNum()
        if node_index not in connected_atom_list:
            continue

        atoms_feature[connected_atom_map[node_index]] = atomicnum  # atom nodes indexed by atomicnum

        if symbol in data1.entities:
            attribute_id = [h for (r, h) in data1.t2rh[data1.entity2id[symbol]]]
            rid = [r for (r, h) in data1.t2rh[data1.entity2id[symbol]]]  # relation ids

            begin_attributes.extend(attribute_id)  # add attribute ids
            end_atoms.extend([node_index] * len(attribute_id))  # add atom ids
            rel_features.extend(
                i + 4 for i in rid)  # first 4 ids are prepared for bonds, relation ids begin after bond ids
        for line in lines:
            patt = line.split(" ", 1)[0]
            pat = Chem.MolFromSmiles(patt)
            if mol.HasSubstructMatch(pat):
                attribute_id = [h for (r, h) in data2.t2rh[data2.entity2id[patt]]]
                rid = [r for (r, h) in data2.t2rh[data2.entity2id[patt]]]  # relation ids

                begin_attributes.extend(attribute_id)  # add attribute ids
                end_atoms.extend([node_index] * len(attribute_id))  # add atom ids
                rel_features.extend(
                    i + 4 for i in rid)


                # get list of attribute ids and features
    if begin_attributes:
        attribute_id = sorted(list(set(begin_attributes)))
        node_id = [i + len(connected_atom_list) for i in range(len(attribute_id))]
        attrid2nodeid = dict(zip(attribute_id, node_id))  # dict: attribute_id in triples --> node_id in dglgraph
        nodeids = [attrid2nodeid[i] for i in begin_attributes]  # list of attribute ids

        nodes_feature = [i + 118 for i in
                         attribute_id]  # first 118 ids are prepared for atoms, attribute ids begin after atom ids

    # get list of atom ids and bond features
    begin_indexes = []
    end_indexes = []
    bonds_feature = []
    edge_type = []

    for bond in mol.GetBonds():
        bond_feature = bondtype_features(bond)

        begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        bonds_feature.append(bond_feature)

        begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        bonds_feature.append(bond_feature)
    edge_type.extend([0] * len(bonds_feature))

    # add ids and features of attributes and relations
    if end_atoms:
        begin_indexes.extend(nodeids)
        end_indexes.extend(end_atoms)
        atoms_feature.extend(nodes_feature)
        bonds_feature.extend(rel_features)
        edge_type.extend([1] * len(rel_features))

    # create dglgraph
    graph = dgl.graph((begin_indexes, end_indexes), idtype=torch.int32)
    graph.edata['e'] = torch.tensor(bonds_feature, dtype=torch.long)
    graph.ndata['h'] = torch.tensor(atoms_feature, dtype=torch.long)
    graph.edata['etype'] = torch.tensor(edge_type, dtype=torch.long)  # 0 for bonds & 1 for rels
    return graph

def drug2emb_encoder(x):

        ## Sequence encoder parameter
        vocab_path = './ESPF/drug_codes_chembl.txt'
        bpe_codes_drug = codecs.open(vocab_path)
        dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
        sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
        idx2word_d = sub_csv['index'].values
        words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
        max_d = 50
        t1 = dbpe.process_line(x).split()
        try:
            i1 = np.asarray([words2idx_d[i] for i in t1])
        except:
            i1 = np.array([0])
            print('error:', x)

        l = len(i1)

        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l))

        else:
            i = i1[:max_d]
            input_mask = [1] * max_d
        return i, np.asarray(input_mask)

def smile_to_graph(smile):
        molecule = Chem.MolFromSmiles(smile)
        n_atoms = molecule.GetNumAtoms()
        atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]
        adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
        node_features = np.array([atom_features(atom) for atom in atoms])
        n_edge_features = 4
        edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
            edge_features[i, j, bond_type] = 1
            edge_features[j, i, bond_type] = 1

        return adjacency, node_features, edge_features

BONDTYPE_TO_INT = defaultdict(
        lambda: 0,
        {
            BondType.SINGLE: 0,
            BondType.DOUBLE: 1,
            BondType.TRIPLE: 2,
            BondType.AROMATIC: 3
        }
    )

def molgraph_collate_fn(data):
    n_samples = len(data)
    (adj_1, node_fts_1, edge_fts_1),targets_0,d1,mask_1,bg,entity_emb, relation_emb= data[0]
    n_nodes_largest_graph_1 = max(map(lambda sample: sample[0][0].shape[0], data))

    n_node_fts_1 = node_fts_1.shape[1]
    n_edge_fts_1 = edge_fts_1.shape[2]

    n_targets = 1
    n_emb= d1.shape[0]
    n_mask=mask_1.shape[0]

    adjacency_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_nodes_largest_graph_1)
    node_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_node_fts_1)
    edge_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_nodes_largest_graph_1, n_edge_fts_1)

    target_tensor = torch.zeros(n_samples, n_targets)
    d1_emb_tensor=torch.zeros(n_samples, n_emb)
    mask_1_tensor=torch.zeros(n_samples, n_mask)

    for i in range(n_samples):
        (adj_1, node_fts_1, edge_fts_1), target, d1, mask_1, bg,entity_emb, relation_emb= data[i]
        n_nodes_1 = adj_1.shape[0]

        adjacency_tensor_1[i, :n_nodes_1, :n_nodes_1] = torch.Tensor(adj_1)
        node_tensor_1[i, :n_nodes_1, :] = torch.Tensor(node_fts_1)
        edge_tensor_1[i, :n_nodes_1, :n_nodes_1, :] = torch.Tensor(edge_fts_1)

        target_tensor[i] = torch.tensor(target)
        d1_emb_tensor[i] = torch.IntTensor(d1)
        mask_1_tensor[i] = torch.tensor(mask_1)

    return adjacency_tensor_1, node_tensor_1, edge_tensor_1,target_tensor,d1_emb_tensor,mask_1_tensor,bg,entity_emb, relation_emb
