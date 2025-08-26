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
    input_conf = OmegaConf.load(args.config)
    config = OmegaConf.merge(basic_config, input_conf)
    logger.info(config)

    # load data 
    train_data, test_data = load_data(train_path = config.data.train_path,
                    train_label_path = config.data.train_label_path,
                    test_path = config.data.test_path,
                    test_label_path = config.data.test_label_path,
                    gene_selection = config.model.feature_size,
                    class_cond=config.model.class_cond,
                    gene_set = args.gene_set,
                    data_filter=config.data.filter,
    )
   
    data = test_data if args.valid else train_data
    # umap plot 
    showdata(data,
            dir = get_blob_logdir(),
            schedule_plot = "stage",
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
    parser.add_argument("--config", type=str, default="configs/random/mrna_8.yaml")
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--gene_set", type=str, default="Random")
    parser.add_argument("--random", action='store_true')
    parser.add_argument("--valid", action='store_true')
    args = parser.parse_args()
    main(args)