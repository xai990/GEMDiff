""" 
Umap plot experiments with different gene set 
"""

import argparse
from omegaconf import OmegaConf
from diffusion import logger, dist_util
from diffusion.datasets import load_data, data_loader, read_file
import torch.distributed as dist
import torch as th 
import numpy as np 
from diffusion.train_util import get_blob_logdir
import datetime
import os 
import pandas as pd 


def main(args):
    logger.configure(dir=args.dir)
    logger.log("**********************************")
    logger.log("log configure")
    dist_util.setup_dist()
    logger.log(f"device information:{dist_util.dev()}")
    basic_config = create_config()
    config = OmegaConf.create(basic_config)
    logger.info(config)

    df = pd.read_csv(config.data.data_path, sep='\t', index_col=0)

    
    logger.log("gene selection complete...")
    if args.gene_set.lower().endswith('.tsv'):
        geneset = pd.read_csv(args.gene_set, delimiter='\t')
        geneset = list(set(geneset["#node1"]))
        df = df[df.columns.intersection(geneset)]
    else: # else include gmt and pure -- need to specific the case in the future to avoid protential error 
        geneset = read_file(args.gene_set)
        logger.log(f"The input gene set is {geneset['gene_set']}, contains {geneset['genes']} genes")
        df = df[df.columns.intersection(geneset['genes'])]
    logger.log(f"loaded selected data has {df.shape[1]} genes, {df.shape[0]} samples")

    gene_set_name = geneset['gene_set']    
    output_dir = "coregene/corelist"
    os.makedirs(output_dir, exist_ok=True)
    intersection_list = list(df.columns.intersection(geneset['genes']))
    subset_size = config.data.gene_select
    np.random.seed(1234)
    idx= np.random.choice(range(0,len(intersection_list)), len(intersection_list), replace=False)
    max_idx= len(idx) - (len(idx) % 16)
    for i in range(0, max_idx, subset_size):
        subset_indices = idx[i:i + subset_size]  # Get actual indices in this slice
        subset = [intersection_list[j] for j in subset_indices]
        output_file = os.path.join(output_dir, f'{gene_set_name}_subset_{i//subset_size + 1}')
        with open(output_file, 'w') as out_f:
            line = gene_set_name + '\t' + '\t'.join(subset) + '\n'
            out_f.write(line)

def create_config():
    
    defaults = {
        "data":{
            "data_path": "datasets/breast_train_GEM_transpose.txt",
            "cond": True,
            "gene_select": 16,
        },
        "umap":{
            "n_neighbors": 90,
            "min_dist": 0.3,
        },
    }
    # defaults.update(model_and_diffusion_defaults())
    return defaults 


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="configs/mrna_8.yaml")
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--gene_set", type=str, default="Random")
    args = parser.parse_args()
    main(args)