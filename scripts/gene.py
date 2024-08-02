""" 
Umap plot experiments with different gene set 
"""

import argparse
from omegaconf import OmegaConf
from diffusion import logger, dist_util
from diffusion.datasets import load_data, data_loader
from diffusion.script_util import (
    model_and_diffusion_defaults,
    showdata,
    create_model_and_diffusion,
    get_silhouettescore,
)
import torch.distributed as dist
import torch as th 
import numpy as np 
from diffusion.train_util import get_blob_logdir
import datetime
import os 


def main(args):
    logger.configure(dir=args.dir)
    logger.log("**********************************")
    logger.log("log configure")
    dist_util.setup_dist()
    logger.log(f"device information:{dist_util.dev()}")
    basic_config = create_config()
    config = OmegaConf.create(basic_config)
    logger.info(config)

    # load data 
    train, test = load_data(data_dir = config.data.data_dir,
                gene_selection = config.data.gene_select,
                class_cond=config.data.cond,
                gene_set = args.gene_set,
                random=args.random,
    )
    score = get_silhouettescore(train,n_neighbors = config.umap.n_neighbors,min_dist = config.umap.min_dist,balance =args.balance)
    experiment = args.gene_set if args.gene_set else "Random"
    logger.log(f"{experiment} experiemnt of silhouette score: {score}")
    # umap plot 
    showdata(train,
            dir = get_blob_logdir(),
            schedule_plot = "balance" if args.balance else "origin",
            n_neighbors = config.umap.n_neighbors,
            min_dist = config.umap.min_dist,
            gene_set = args.gene_set,
    )
    
    logger.log("gene selection complete...")


def create_config():
    
    defaults = {
        "data":{
            "data_dir": "datasets",
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
    parser.add_argument("--balance", action='store_true')
    parser.add_argument("--random", action='store_true')
    args = parser.parse_args()
    main(args)