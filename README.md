# GEMDiff: A diffusion model bridge between normal and tumor Gene Expression Matrix

This repository contains the code for the diffusion model and a neural network model for a breast cancer study case. This input files for this datset will be provided soon.
The results can be found on our [paper](https://academic.oup.com/bib/article/26/2/bbaf093/8069412?utm_source=advanceaccess&utm_campaign=bib&utm_medium=email)/[website](https://xai990.github.io/)

## Installation 
GEMDiff is a collection of Python scripts. Recommendation are for running the diffusion model on [Palmetto2](https://www.palmetto.clemson.edu/palmetto/) -- a Clemson University research cluster. To use the Python scripts directly, clone this repository.  All of the Python dependencies can be installed in an Anaconda environment:
```bash
# load Anaconda module if needed 
module load anaconda3/2023.09-0

# clone repository
git clone https://github.com/xai990/DGEMDiff.git

cd GEMDiff
# create conda environment called "GEMDiff"
conda env create -f environment.yml -n GEMDiff

# activate the created conda environment
source activate GEMDiff

# install the package
pip install -e . 

```

## Preparing Input Gene Expression Matrix (GEM) and Group Label File
The user provides training and test gene expression matrices (GEM) and sample_id > group_id label files.  These tab-delimited text files file are stored in the default folder (`"datasets"`). For creating/inputting your own dataset, simply format the GEM into a plain-text file with rows being samples and columns being gene identifiers. Values in each row should be separated by tabs. If the data repository has a different name, modify it in the config file correspondingly. 
```
	Gene1	Gene2	Gene3	Gene4
Sample1	0.523	0.991	0.421	0.829
Sample2	8.891	7.673	3.333	9.103
Sample3	4.444	5.551	6.102	0.013
```
The labels file should contain a label for each sample, corresponding to something such as a condition or phenotype state for the sample. This file should contain two columns, the first being the sample names and the second being the labels. Values in each row should be separated by tabs.

```
Sample1	Label1
Sample2	Label2
Sample3	Label3
Sample4	Label4
```

The gene set list is optional. The file should contain the name and genes for a gene set. The file could be a GMT file or a plain format. Values on each row should be separated by tabs.
```
GeneSet1	Gene1	Gene2	Gene3
```

## Preparing Config File
The config file is in YAML format and contains four stanzas: (1) data: GEM, label, directory paths, locations, (2) model: model architecture hyperparameters, (3) diffusion: diffusion process hyperparameters, (4) train: training hyperparameters.  The default hyperparameters hard-coded in the train.py script can be overridden in the config file.  Here is an example config file.  
```
data:
  dir_out: "results"
  train_path: "datasets/KIDN_KIRP.train"
  train_label_path: "datasets/KIDN_KIRP.train.label"
  test_path: "datasets/KIDN_KIRP.test"
  test_label_path: "datasets/KIDN_KIRP.test.label"
  filter: null #change to 'replace' to remove 'NA' values 
  corerate: 1 
  
model:
  class_cond: True
  dropout: 0.0
  n_layer: 4
  n_head: 2
  feature_size: 21
  
diffusion:
  noise_schedule: "cosine"
  diffusion_steps: 1000
  log_every_t: 10
  learn_sigma: False
  
train:
  lr: 0.00003
  # num_epoch: 1
  batch_size: 16
  schedule_plot: False
  # log_interval: 100
  # save_interval: 1
```

## Colab example
Colab Examples: We provide google colabs to help reproduce and customize our repo, which includes experiments train and perturb and visualization.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ssJGavsFgiFfMMgVfHXz1YXf0HFzZS5G?usp=sharing)

## Training the Diffusion Model
To train your model, there are some hyperparameters with default values in the train.py script. Hyperparameters are split into three groups: model architecture ("model:" in config file), diffusion process ("diffusion:" in config file), and training ("train:" in config file) flags. We include some reasonable defaults for baseline [config files](configs). The default hyperparameters hard-coded in the train.py script can be overridden in the config file.  The hyperparameters will be stored in the trained model file (e.g. model10000.pt stored in the log directory). Once you have set up your hyperparameters, you can run an experiment like this:

```
python scripts/train.py --config "<config file path>" --dir "<log directory path>"
```
`--gene_set` is an optional input for gene set list, defualt as "Random". The above training script saves checkpoints to .pt files in the logging directory. These checkpoints will have names like model40000.pt, which stores the learnable parameters both of models and EMAs.

## Perturbing Gene Expression Matrices 
The perturbing process need to load from checkpoints. The default setting is to sample from the EMAs (Exponential Moving Averages), since those produce much better transformation. 
```
python scripts/pertub.py --config "<config file path>" --dir "<log directory path> --model_path "<model path> --valid" 
```
`--gene_set` is an optional input for gene set list, defualt as "Random".
`--valid` is to valid model with test dataset. 

## Gene Expression Matrix Augmentation from Trained Model
The default setting is to sample from the EMAs (Exponential Moving Averages), since those produce much better samples. 
```
python scripts/sample.py --model_path "<pt file path>"  --dir "<log directory path>" --dir_out "<output directory path>" 
```

## Visualizing the Samples 
The plotting script visualizes data by UMAP plot and assign the silhouette score as the cluster quality. 
```
python scripts/gene.py --config "<config file path>" --dir "<log directory path>"  
```
`--gene_set` is an optional input for gene set list, default as "Random".
`--balance` is to set the sample number of each label data shown on the figure. 
`--random` is to assign different seeds each time running the script. 
`--valid` is to plot the test dataset. 

The detailed descriptions about the config parameter are as following:
| Parameter name | Description of parameter |
| --- | --- |
| config.data.train_path                | Path to the training dataset file                                                                   |
| config.data.train_label_path          | Path to the training dataset labels                                                                 |
| config.data.test_path                 | Path to the test dataset file                                                                       |
| config.data.test_label_path           | Path to the test dataset labels                                                                     |
| config.data.filter                    | Data filtering option: set to 'replace' to substitute 'NA' values, null for no filtering            |
| config.data.corerate                  | Number of standard deviations from the perturbation mean to filter data                             |
| config.model.class_cond               | Enable class conditioning in the model                                                              |
| config.model.dropout                  | Dropout rate for regularization (0.0 means no dropout)                                              |
| config.model.n_layer                  | Number of layers in the model architecture                                                          |
| config.model.n_head                   | Number of attention heads in each layer                                                             | 
| config.model.feature_size             | Dimension size of feature embeddings   (has to be set as same as dimension size of gene features)   |
| config.diffusion.noise_schedule       | Type of noise schedule for diffusion process (defaults to `cosine`)                                 |
| config.diffusion.diffusion_steps      | Total number of steps in the diffusion process (defaults to `1000`)                                 |
| config.diffusion.log_every_t          | Log data every N diffusion timesteps (defaults to `10`)                                             |
| config.diffusion.learn_sigma          | If True, model learns the variance parameter; if False, variance is fixed                           |
| config.train.lr                       | Learning rate for optimizer during training (defaults to `3e-5`)                                    |
| config.train.num_epoch                | Number of training epochs (defaults to `10001`)                                                     |
| config.train.batch_size               | Number of samples processed in each training batch (defaults to `32`)                               |
| config.train.schedule_plot            | If True, save the schedule plots during the diffusion process                                       |
| config.train.log_interval             | How often to log training metrics (defaults to `100`)                                               |
| config.train.save_interval            | How often to save model checkpoints (defaults to `10000`)                                           |




## <span id="citelink">Citation</span>
If you find this repository useful in your research, please consider citing our paper:

```
@article{10.1093/bib/bbaf093,
    author = {Ai, Xusheng and Smith, Melissa C and Feltus, F Alex},
    title = {GEMDiff: a diffusion workflow bridges between normal and tumor gene expression states: a breast cancer case study},
    journal = {Briefings in Bioinformatics},
    volume = {26},
    number = {2},
    pages = {bbaf093},
    year = {2025},
    month = {03},
    issn = {1477-4054},
    doi = {10.1093/bib/bbaf093},
    url = {https://doi.org/10.1093/bib/bbaf093},
    eprint = {https://academic.oup.com/bib/article-pdf/26/2/bbaf093/62374997/bbaf093.pdf},
}
	
```







