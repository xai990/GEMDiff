# GEMDiff: A diffusion model bridge between normal and tumor Gene Expression Matrix

This repository contains the code for the diffusion model and a neural network model for a breast cancer study case. 
## Installation 

GEMDiff is a collection of Python scripts. Recommand that run diffusion model on [Palmetto](https://www.palmetto.clemson.edu/palmetto/) -- a Clemson university research cluster. To use the Python scripts directly, clone this repository.  All of the Python dependencies can be installed in an Anaconda environment:
```bash
# load Anaconda module if needed 
module load anaconda3/2023.09-0

# clone repository
git clone https://github.com/xai990/DGEMDiff.git

cd GEMDiff
# create conda environment called "GEMDiff"
conda env create -f environment.yml

# activate the created conda environment
source activate GEMDiff

# install the package
pip install -e . 

```


## Perparing data
The training code reads gene expression matrix from a directory. The default folder(`"datasets"`), include training and testing GEM files and corresponding label files. 
For creating/inputting your own dataset, simply format the GEM into a plain-text file with rows being samples and columns being genes. Values in each row should be separeated by tabs. If the data repository has a different name, modify it in the config file correspondingly. 
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

## Training 
To train your model, there are some hyperparameters. We will split up our hyperparameters into three groups: model architecture, diffusion process, and training flags. We include some reasonable defaults for baseline [config files](configs) (You could also setup the paprameter by creating your own config file). Once you have setup your hyper-parameters, you can run an experiment like so:

```
python scripts/train.py --config "<config file path>" --dir "<log directory path>"
```
`--gene_set` is an optional input for gene set list, defualt as "Random". The above training script saves checkpoints to .pt files in the logging directory. These checkpoints will have names like model40000.pt, which stores the learnable parameters both of models and EMAs.

## Perturbing
The perturbing process need to load from checkpoints. The default setting is to sample from the EMAs (Exponential Moving Averages), since those produce much better transformation. 
```
python scripts/pertub.py --config "<config file path>" --dir "<log directory path> --model_path "<model path> --valid" 
```
`--gene_set` is an optional input for gene set list, defualt as "Random".
`--valid` is to valid model with test dataset. 

## Gene Augmentation 
The default setting is to sample from the EMAs (Exponential Moving Averages), since those produce much better samples. 
```
python scripts/sample.py --model_path "<pt file path>"  --dir "<log directory path>" --dir_out "<output directory path>" 
```


## Plotting and qualifying cluster 
The plotting script visulizes data by UMAP plot and assign the silhouette score as the cluster quaility. 
```
python scripts/gene.py --config "<config file path>" --dir "<log directory path>"  
```
`--gene_set` is an optional input for gene set list, defualt as "Random".
`--balance` is to set the sample number of each label data shown on the figure. 
`--random` is to assign different seeds each time running the script. 
`--vaild` is to plot the test dataset. 

