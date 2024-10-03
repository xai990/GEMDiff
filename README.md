# DDPM-mRNA-augmentation-light
# DDPM for breast cancer mRNA data augmentation 

This repository contains the code for the diffusion model and a neural network model, specifically for breast cancer. 
## Installation 

DDPM is a collection of Python scripts. Recommand that run diffusion model on [Palmetto](https://www.palmetto.clemson.edu/palmetto/) -- a Clemson university research cluster. To use the Python scripts directly, clone this repository.  All of the Python dependencies can be installed in an Anaconda environment:
```bash
# load Anaconda module if needed 
module load anaconda3/2022.05-gcc/9.5.0 

# clone repository
git clone https://github.com/xai990/DDPM-mRNA-augmentation.git

cd DDPM-mRNA-augmentation
# create conda environment called "DDPM"
conda env create -f environment.yml

# install the package
python setup.py install 

# run DDPM on breast cancer data  -- make sure the scripts are running on a computed node (an interactive job includes at least one gpu)
python scripts/main.py

```
