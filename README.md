# C-FREE

**Official repository for** [**"Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining"**](https://arxiv.org/abs/2509.22468), accepted at ICML 2026. 

## Installation

To install the required dependencies, use the following commands:

```bash
conda create -n cfree python=3.10
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install ogb rdkit schedulefree wandb multimethod matplotlib
pip install flash-attn --no-build-isolation
```

## Experiments
### Pre-training Experiments

To run the pre-training pipeline, we use the [GEOM dataset](https://github.com/learningmatter-mit/geom). 

**Note:** We rename the dataset files to `geom_*` to differentiate them from the MARCEL Drugs dataset. GEOM contains the full dataset (~304k molecules), while MARCEL Drugs is a subset (~75k molecules).

1. Download the `rdkit_folder.tar.gz` archive from [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF). 
2. Extract the entire archive
3. Rename `drugs_summary.json` to `summary_geom.json`
4. Place the extracted files (including `summary_geom.json` and all the pickle files) in: `./data/raw_data/geom/raw/`


### MARCEL Experiments

To run the MARCEL experiments:

1. Download the `Drugs.zip` and `Kraken.zip` archives from the [official MARCEL repo](https://github.com/SXKDZ/MARCEL).
2. Place them in the following locations:
   - `./data/raw_data/drugs/raw/Drugs.zip`
   - `./data/raw_data/kraken/raw/Kraken.zip`

## Running Experiments

Experiment configuration files are available in the `./cfgs` folder. To run an experiment using a specific configuration file run

```bash
python pretrain.py --cfg ./cfgs/pretrain-default.yaml
```
or

```bash
python finetune.py --cfg ./cfgs/finetune-default.yaml
```

Alternatively, you can define multiple setups within the `main.py` script and run experiments using different datasets or different seeds with: 

```bash
python main.py 
```

## Codebase

This work is based on [MolMix](https://github.com/andreimano/MolMix/).

This repository includes modified implementations from:
- [SchNet](https://github.com/atomistic-machine-learning/SchNet)
- [PaiNN](https://github.com/atomistic-machine-learning/schnetpack)
- [ESAN](https://github.com/beabevi/ESAN)

Additionally, various parts of the code are adapted from the [MARCEL repository](https://github.com/SXKDZ/MARCEL).

## Citation

If you build on this work, please cite the following:

```bibtex
@misc{ariguib2025learningneighborhoodcontrastfreemultimodal,
      title={Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining}, 
      author={Boshra Ariguib and Mathias Niepert and Andrei Manolache},
      year={2025},
      eprint={2509.22468},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.22468}, 
}
```

```bibtex
@misc{manolache2024molmix,
      title={MolMix: A Simple Yet Effective Baseline for Multimodal Molecular Representation Learning}, 
      author={Andrei Manolache and Dragos Tantaru and Mathias Niepert},
      year={2024},
      booktitle={Machine Learning for Structural Biology Workshop, NeurIPS 2024},
      url={https://arxiv.org/abs/2410.07981}, 
}
```