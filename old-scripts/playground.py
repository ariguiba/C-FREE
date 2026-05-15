from rdkit.Chem import Draw
import torch
from rdkit import Chem
import torch
from preprocessing.dataloaders import get_data
from types import SimpleNamespace
import itertools

from preprocessing.datasets.chiro import get_chiro_dataset
from preprocessing.subgraphs import policy2transform
from preprocessing.utils.create_subgraphs import create_ego_pairs, create_fragment_pairs, create_full_pairs, create_murcko_pairs

# from datasets.datasets import get_data
# from layers.ssl_models import SimpleGIN

class Args:
    def __init__(self):
        self.dataset = "geom"
        self.n_conformers = 1
        self.with_3d = False
        self.debug = False
        self.simple = False
        # self.data = SimpleNamespace(hidden_dim = 4, n_gnn_layers = 2)
        self.policy = 'original'
        self.ssl = True
            

# Testing one molecule
def plot_one_molecule(data, args):
    mol = data.mol[0]
    img = Draw.MolToImage(mol)
    img.save(f"exp-mol-{args.dataset}.png")

# # Getting the count of all the different rings/paths
# def get_fragments_count(loader, args):
#     print(f"Number of molecules in the dataset: {len(loader.dataset)}")
#     vocab_size = 50
#     final_sum = torch.zeros(vocab_size)
#     for data in loader:
#         flat = data.y_count # requires PreTransform: CountSubstructures 
#         flat = flat.view(-1) # Make sure it's a 1D tensor
#         num_molecules = flat.shape[0] // vocab_size
#         reshaped = flat.view(num_molecules, vocab_size)
#         summed = reshaped.sum(dim=0)  # Shape: [vocab_size]
#         final_sum += summed
#     return final_sum

# def draw_fragments_count(count):
#     frequency_map = { }
#     max_ring = 10
#     for i in range(len(count)):
#         if i < max_ring:
#             frequency_map[f'[0, {i+3}]'] =  count[i]
#         else:
#             frequency_map[f'[1, {i-max_ring+2}]'] = count[i] 

#     print(frequency_map)

#     # --- Generate Ring Fragments ---
#     ring_mols = []
#     ring_labels = []
#     for ring_size in range(4, 11):  # 4- to 10-membered rings
#         smiles = 'C' + ('1' + 'C'*(ring_size-2) + '1')  # e.g., C1CCC1, C1CCCC1, ...
#         mol = Chem.MolFromSmiles(smiles)
#         ring_mols.append(mol)
#         ring_labels.append(f"[0, {ring_size-1}]")

#     # --- Generate Chain Fragments ---
#     chain_mols = []
#     chain_labels = []
#     for chain_length in range(2, 11):  # 1- to 10-carbon chains
#         smiles = 'C' * chain_length
#         mol = Chem.MolFromSmiles(smiles)
#         chain_mols.append(mol)
#         chain_labels.append(f"[1, {chain_length}]")

#     # --- Combine and Draw ---
#     all_frags = ring_mols + chain_mols
#     #all_labels = ring_labels + chain_labels
#     all_labels = []
#     for label in ring_labels + chain_labels:
#         count = frequency_map.get(label, 0)
#         full_label = f"{label} Freq: {count}"
#         all_labels.append(full_label)

#     img = Draw.MolsToGridImage(all_frags, molsPerRow=7, subImgSize=(200,200), legends=all_labels)
#     img.save(f"{args.dataset}-counter.png")
    

    
# def fragmenter(args):
#     _, _, tst_loader, _ = get_data(args)

#     # Plot one molecule
#     data = next(iter(tst_loader.loader))

    
#     # plot_one_molecule(first_data_frag, args)

#     # Pass data through our simple GIN
#     model = SimpleGIN(args)
#     out = model(data)
#     print(out)

#     # Counts the different fragments in the test loader
#     # final_sum = get_fragments_count(tst_loader.loader, args)
#     # torch.set_printoptions(sci_mode=False)
#     # print(final_sum)
    
#     # # Visualize and save the fragments count   
#     # draw_fragments_count(final_sum)

# def load_model():
#     file = ""
#     checkpoint = torch.load(f"checkpoints/{file}.pth", weights_only=True)
#     print(checkpoint)

def create_subgraphs():
    import torch
    from preprocessing.subgraphs import policy2transform
    from preprocessing.datasets.moleculenet import MyMoleculeNet
    import tqdm

    # Path to your processed files
    for dataset_name in [ 'hiv', 'muv']:
    
        path = f"data/datasets/{dataset_name}"
        new_pre_transform = policy2transform(
                policy='subgraphs',
                subgraph_types=['3-ego'],
                num_samples=[1], 
                process_subgraphs=lambda x: x,
                pbar=iter(tqdm.tqdm(range(200000))),
                with_3d=True,
            )

        # Load the processed dataset
        dataset = MyMoleculeNet(
            root=f"{path}/{dataset_name}_3d_1_original/", #{dataset}/processed/data.pt", 
            name=f"{dataset_name}"
        )

        # Access the `data` and `slices`
        data, slices, _ = torch.load(dataset.processed_paths[0])

        # Iterate over individual graphs
        new_data_list = []
        for i in range(len(dataset)):
            g = dataset[i]               # reconstruct one graph
            g = new_pre_transform(g)     # apply your new pre_transform
            new_data_list.append(g)

        new_data, slices = dataset.collate(new_data_list)

        new_dataset = MyMoleculeNet(
            root=f"{path}/{dataset_name}_3d_1_mix_1k-3-ego/",
            name=f"{dataset_name}"
        )

        # Save back
        torch.save((new_data, slices), new_dataset.processed_paths[0]) 

def subgraphs_features():
    args = SimpleNamespace()
    args.data = Args()
    trn_loader, _, tst_loader, _ = get_data(args)
    loader = trn_loader.loader

    samples = list(itertools.islice(loader, 5))

    results = []
    for sample in samples:
        three_subgraphs = []
        group_counter = 0
        for _ in range(3):
            subgraphs, group_counter = create_ego_pairs(
                data=sample,
                group_counter=group_counter,
                num_hops=1,
                generate_all=False,
                k=1,
                with_3d=False,
            )
            three_subgraphs.append(subgraphs)
            print(subgraphs)
        results.append(three_subgraphs)
        print("-----")

    for sample_idx, three_subgraphs in enumerate(results):
        original = samples[sample_idx]
        print(f"\n--- Sample {sample_idx} (original: {original.num_nodes} atoms) ---")
        
        for call_idx, subgraphs in enumerate(three_subgraphs):
            print(f"  Call {call_idx + 1}:")
            for sg in subgraphs:
                num_atoms = sg.num_nodes
                num_bonds = sg.edge_index.shape[1] // 2
                avg_degree = num_bonds / num_atoms if num_atoms > 0 else 0
                coverage = num_atoms / original.num_nodes

                print(f"    atoms={num_atoms}, bonds={num_bonds}, "
                    f"avg_degree={avg_degree:.2f}, coverage={coverage:.2%}")

    # for i, _ in enumerate(samples):
    #     print(samples[i])

    import pandas as pd

    rows = []

    for sample_idx, three_subgraphs in enumerate(results):
        original = samples[sample_idx]
        original_num_bonds = original.edge_index.shape[1] // 2
        original_avg_degree = original_num_bonds / original.num_nodes if original.num_nodes > 0 else 0

        for call_idx, subgraphs in enumerate(three_subgraphs):
            context_subgraphs = subgraphs[::2]

            for sg in context_subgraphs:
                num_atoms = sg.num_nodes
                num_bonds = sg.edge_index.shape[1] // 2
                avg_degree = num_bonds / num_atoms if num_atoms > 0 else 0
                coverage = num_atoms / original.num_nodes

                rows.append({
                    'sample_idx': sample_idx,
                    'call_idx': call_idx + 1,
                    'original_num_atoms': original.num_nodes,
                    'original_avg_degree': round(original_avg_degree, 3),
                    'num_atoms': num_atoms,
                    'num_bonds': num_bonds,
                    'avg_degree': round(avg_degree, 3),
                    'coverage': round(coverage, 3),
                })

    df = pd.DataFrame(rows)

    print(df.to_string(index=False))
    print("\n--- Summary Statistics ---")
    print(df[['original_num_atoms', 'original_avg_degree', 'num_atoms', 'num_bonds', 'avg_degree', 'coverage']].describe().round(3))


if __name__ == "__main__":
    # create_subgraphs()
    # subgraphs_features()
    args = SimpleNamespace(data_path = "data/raw_data")
    args.data = Args()
    trn_df, _, tst_df, _ = get_chiro_dataset(args, force_subset=False) #get_data(args)
    print(trn_df)