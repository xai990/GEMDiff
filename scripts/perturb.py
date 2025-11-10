"""
This script performs perturbation analysis using a trained diffusion model. 

It loads a pre-trained diffusion model, perturbs gene expression data (either training or testing data) from one class to another,
visualizes the perturbed data, and identifies the most perturbed genes.

The script relies on configuration files to specify data paths, model parameters, and other settings.

Usage:
    python script_name.py --config configs/your_config.yaml --model_path path/to/your/model.pth --dir path/to/log/dir [--valid] [--gene_set your_gene_set]

    --config: Path to the configuration YAML file.
    --model_path: Path to the pre-trained model checkpoint.
    --dir: Directory to save logs and results.
    --valid: Use the validation (test) dataset instead of the training dataset.
    --gene_set: The gene set to use for filtering.
"""

import argparse
from omegaconf import OmegaConf
from diffusion import logger, dist_util
from diffusion.datasets import (load_data, 
    data_loader, 
    sample_screen,
    LabelGeneDataset,
    balance_sample
)
from diffusion.script_util import (
    model_and_diffusion_defaults,
    showdata,
    create_model_and_diffusion,
    filter_gene,
)
import torch.distributed as dist
from diffusion.resample import create_named_schedule_sampler
import torch as th 
import numpy as np 
from diffusion.train_util import get_blob_logdir
import os 
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import OrderedDict
import datetime

def main(args):
    #args = create_argparser().parse_args()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logger.configure(dir=args.dir)
    logger.log("**********************************")
    logger.log("log configure")
    # Set up distributed training environment
    dist_util.setup_dist()
    logger.log(f"device information:{dist_util.dev()}")
    logger.log("Load the config...")
    # Create and merge configuration files
    basic_conf = create_config()
    input_conf = OmegaConf.load(args.config)
    config = OmegaConf.merge(basic_conf, input_conf)
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
    classes = train_data.classes
    logger.log(f"Available classes in dataset: {classes}")
    assert len(classes) > 1, ("The class is not enough for the perturb experiment")
    source_class = config.perturb.source_class if config.perturb.source_class else classes[1]
    target_class = config.perturb.target_class if config.perturb.target_class else classes[0]
    if source_class not in classes:
        raise ValueError(f"Source class '{source_class}' not found in dataset classes: {classes}")
    if target_class not in classes:
        raise ValueError(f"Target class '{target_class}' not found in dataset classes: {classes}")
    source_idx = classes.index(source_class)
    target_idx = classes.index(target_class)
    # separate the tumor and normal data
    train_N, train_T = sample_screen(train_data,source_idx,target_idx)
    test_N, test_T = sample_screen(test_data,source_idx,target_idx)
    # Set model configuration parameters
    logger.log("creating model and diffusion ... ")
    logger.info(f"The model feature size is : {config.model.feature_size}")
    config.model.patch_size = config.model.feature_size
    config.model.n_embd = config.model.patch_size * 8
    logger.info(config)
    model_config = OmegaConf.to_container(config.model, resolve=True)
    diffusion_config = OmegaConf.to_container(config.diffusion, resolve=True)
    # Create model and diffusion process
    model, diffusion = create_model_and_diffusion(**model_config, **diffusion_config)
    ema= deepcopy(model)
    # Load model weights from checkpoint (using EMA weights)
    state_dict = th.load(args.model_path, weights_only=False)
    ema.load_state_dict(state_dict["ema"])
    ema.to(dist_util.dev())
    ema.eval()

    logger.log(f"pertubing the source {source_class} label to target {target_class} label")
    source_label = {}
    if config.model.class_cond:
        source_label['y'] =  th.full((test_T.shape[0] if args.valid else train_T.shape[0],), source_idx, device=dist_util.dev(), dtype=th.int32)
    noise_T = diffusion.ddim_reverse_sample_loop(
        ema,
        test_T.shape if args.valid else train_T.shape,
        th.Tensor(test_T).float().to(dist_util.dev()) if args.valid else th.Tensor(train_T).float().to(dist_util.dev()),
        clip_denoised=False,
        model_kwargs=source_label,
    )
    target_label = {}
    if config.model.class_cond:
        target_label['y'] =  th.full((test_T.shape[0] if args.valid else train_T.shape[0],), target_idx, device=dist_util.dev(), dtype=th.int32) 
    target = diffusion.ddim_sample_loop(ema,noise_T.shape,noise= noise_T,clip_denoised=False,model_kwargs=target_label)
    shape_str = "x".join([str(x) for x in noise_T.shape])
    out_path = os.path.join(get_blob_logdir(), f"reverse_sample_{shape_str}.npz")
    target = target.cpu().numpy()
    np.savez(out_path, target)
    logger.log(f"saving perturb data array to {out_path}")
    logger.log("visulize the perturbed data and real data")
    
    plotdata = [test_N, test_T, target] if args.valid else [train_N, train_T, target]
    showdata(plotdata,dir = get_blob_logdir(), schedule_plot = "perturb", n_neighbors =config.umap.n_neighbors,min_dist=config.umap.min_dist)
    
    logger.log("filter the perturbed gene -- 1 std")
    gene_index = filter_gene(test_T if args.valid else train_T, target, config.data.corerate)
    if args.valid:
        logger.log(f"The indentified genes are: {test_data.find_gene(gene_index)} -- {config.data.corerate} standard deviation of the perturbation among all {test_N.shape[1]} gene")
    else:
        logger.log(f"The indentified genes are: {train_data.find_gene(gene_index)} -- {config.data.corerate} standard deviation of the perturbation among all {train_N.shape[1]} gene")
    
    logger.log("pertubing complete")



def create_config():
    
    defaults = {
        "data":{
            "data_dir": None,
            "dir_out": "results",
            "gene_selection": None,
            "drop_fraction":0,
        },
        "train":{
            "microbatch": 16,
            "log_interval": 1000,
            "save_interval": 20000,
            "schedule_plot": False,
            "resume_checkpoint": "",
            "ema_rate": 0.9999,
            "num_epoch":40001,
            "schedule_sampler":"uniform",
        },
        "perturb":{
            "source_class": "tumor",
            "target_class": "normal",
        },
        "umap":{
            "n_neighbors":90,
            "min_dist":0.3,
        }

    }
    defaults.update(model_and_diffusion_defaults())
   
    return defaults  


@th.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/random/mrna_16.yaml")
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--gene_set", type=str, default="Random")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--valid", action='store_true')
    args = parser.parse_args()
    main(args)
