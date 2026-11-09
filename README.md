# BMCM

PyTorch implementation of BMCM, the Balanced Multi-modal Clustering Method, from the paper

> Xiao Zhang, Rongcai Zhao, Kai Nie and Yiqiao Mao. *Agree to Disagree: A Balanced Multi-modal Clustering Method via Reliable Contrastive Learning.* Technologies (MDPI), under review.

BMCM weights the contrastive loss of every modality pair by a reliability weight. An evaluator network predicts the weight from intra-modal quality scores and inter-modal relation descriptors, and its training target caps the weight of a pair at the quality of the weaker modality.

## Repository layout

| Path | Content |
| --- | --- |
| `bmcm/data.py` | Loading of `data/<name>.mat` and per-feature min-max scaling to [0, 1] |
| `bmcm/quality.py` | Cluster separability $D_u$, assignment entropy $h_u$, quality score $q_u$, chance-adjusted partition agreement $A_{uv}$, normalized MMD $\widetilde m_{uv}$ |
| `bmcm/model.py` | Modality encoders, shared clustering head, inter-modal reliability evaluator (IRE) |
| `bmcm/losses.py` | Feature-level contrast with cluster-guided negative sampling, cluster-level contrast, DDC loss |
| `bmcm/train.py` | Training, quality-weighted fusion, evaluation and logging |
| `bmcm/metrics.py` | ACC, NMI and ARI |
| `configs/` | One configuration file per dataset, with the hyperparameters and the random seed of the reported run |
| `data/` | Feature matrices of Caltech-3V, Caltech-4V, Caltech-5V and Caltech-6V |
| `scripts/run_main.sh` | The seven runs of Tables 4 and 5 |
| `scripts/run_experiments.py` | The runs of the ablation, descriptor, sensitivity, weight-allocation, pair-intervention, degradation and t-SNE experiments |

## Requirements

```bash
pip install -r requirements.txt
```

The reported runs used two machines:

| Machine | Python | PyTorch | NumPy | SciPy | scikit-learn |
| --- | --- | --- | --- | --- | --- |
| NVIDIA A100 80GB | 3.11.16 | 2.8.0 (CUDA 12.8) | 2.4.6 | 1.17.1 | 1.9.1 |
| NVIDIA GeForce RTX 4090 | 3.13.5 | 2.9.0 (CUDA 12.8) | 1.26.4 | 1.15.3 | 1.6.1 |

## Data

Each dataset is one MATLAB file `data/<name>.mat` with the views `X1`, `X2`, ..., `XV` (one matrix of size samples x dimensions per view) and the labels `Y`. The four Caltech files are included in this repository. They contain 1,400 images of seven Caltech-101 categories described by hand-crafted features:

| File | Dataset | Views (dimension) |
| --- | --- | --- |
| `data/caltech3v.mat` | Caltech-3V | WM (40), CENTRIST (254), LBP (928) |
| `data/caltech4v.mat` | Caltech-4V | WM (40), CENTRIST (254), LBP (928), GIST (512) |
| `data/caltech5v.mat` | Caltech-5V | WM (40), CENTRIST (254), HOG (1984), GIST (512), LBP (928) |
| `data/caltech6v.mat` | Caltech-6V | Gabor (48), WM (40), CENTRIST (254), HOG (1984), GIST (512), LBP (928) |

The features are stored as float32 exactly as they were used in the experiments. Min-max scaling is applied by the code.

YouTube, MS-COCO and VOC2007 are not redistributed here. Their sources are listed in Section 4.1 and in the Data Availability Statement of the paper. To run them, save the features in the same format as `data/youtube.mat` (audio 7169, text 1000, visual 512; 8,557 samples, 30 classes), `data/coco.mat` (image 300, text 300; 61,954 samples, 9 classes) and `data/voc2007.mat` (image 4096, tag 798; 5,011 samples, 20 classes). MATLAB v7.3 files are read with h5py.

## Training

```bash
python -m bmcm.train --config configs/caltech4v.json
```

Any option given on the command line overrides the configuration file, for example `--seed 0 --out runs/caltech4v_s0`. Without `--config`, `--dataset` is required and the defaults of `bmcm/train.py` are used; they equal the values of the configuration files. A run writes to `--out` (default `runs/<dataset>_s<seed>`):

- `config.json`: all options of the run;
- `log.jsonl`: one record per epoch with the loss terms, the pair weights `w`, the descriptors, the quality scores `q`, the fusion coefficients and ACC, NMI and ARI of the fused and of every per-modality prediction;
- `summary.json`: ACC, NMI and ARI at the last epoch, which is the value reported in the paper, and their mean over the last 20 epochs.

## Hyperparameters

All datasets use the same values (Table 3 of the paper):

| Option | Symbol | Value |
| --- | --- | --- |
| `--tau_f` | $\tau_f$, feature-level temperature | 0.1 |
| `--tau_c` | $\tau_c$, cluster-level temperature | 1.0 |
| `--alpha` | $\alpha$, balance in the relation reference $R_{uv}$ | 0.5 |
| `--eps_q` | $\epsilon_q$, constant in the quality score | 0.1 |
| `--eps_f` | $\epsilon_f$, constant in the fusion coefficients | 0.01 |
| `--sigma_q` | $\sigma_q^2$ relative to the median squared distance | 0.15 |
| `--sigma_ddc` | kernel bandwidth of the DDC loss, relative | 0.15 |
| `--fea_dim` | encoder output dimension | 256 |
| `--ire_hidden` | evaluator hidden width | 14 |
| `--lr`, `--batch_size` | Adam learning rate, batch size | 1e-4, 256 |
| `--epochs` | training epochs | 200, MS-COCO 40 |

## Reproducing the paper

```bash
bash scripts/run_main.sh
```

runs every dataset whose data file is present with its configuration file. The seed in each configuration file is the seed of the run reported in Tables 4 and 5. For every dataset BMCM was trained with several seeds and the reported run was selected among them (Section 4.1 of the paper). All other experiments of a dataset use the same seed.

| Dataset | Seed | Machine |
| --- | --- | --- |
| Caltech-3V | 3 | A100, PyTorch 2.8 |
| Caltech-4V | 74 | RTX 4090, PyTorch 2.9 |
| Caltech-5V | 5 | A100, PyTorch 2.8 |
| Caltech-6V | 115 | RTX 4090, PyTorch 2.9 |
| YouTube | 22 | A100, PyTorch 2.8 |
| MS-COCO | 28 | A100, PyTorch 2.8 |
| VOC2007 | 101 | RTX 4090, PyTorch 2.9 |

The code sets every random seed and uses deterministic cuDNN kernels, so a run repeats exactly on the same GPU model and software versions. A different GPU model or PyTorch version changes the floating-point operations and therefore the training trajectory, and the results then vary as they do across seeds.

The other experiments are started with

```bash
python scripts/run_experiments.py <experiment> [--datasets caltech4v,caltech5v] [--dry_run]
```

| Experiment | Paper | Options |
| --- | --- | --- |
| `ablation` | Figure 3, Ablations 1 to 5 | `--weights none`, `--weights uniform`, `--weights learned`, `--levels fea`, `--levels clu` |
| `descriptors` | Table 6 | `--desc_drop sep/ent/agree/mmd`, `--ire_terms qal/cal`, `--qal_reduce mean` |
| `robustness` | Table 7 | `--eps_q`, `--mmd_bw_scale`, `--w_rc`, `--w_ire` |
| `sensitivity` | Figures 7 and 8 | `--alpha`, `--tau_f` |
| `allocation` | Figure 5 | `--alloc uniform_mass`, `--alloc permute` |
| `pairs` | Figure 6 and the leave-one-pair-out utilities | `--weights fixed --fixed_weights w_01,w_02,...` |
| `degradation` | Table 8 | `--corrupt_view 2 --corrupt_type gauss/shuffle --corrupt_level`, uniform baseline `--weights uniform --fusion uniform` |
| `embedding` | Figure 9 | `--save_embedding_epochs 0,49,99,199` |

The pair-weight curves of Figure 4 and the curves of Figure 10 are drawn from the `log.jsonl` files of the main runs. Pairs are ordered as (0,1), (0,2), ..., (1,2), ..., with modalities indexed from 0.
