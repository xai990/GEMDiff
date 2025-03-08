""" 
Train a diffusion model on seq2seq model.
"""

import argparse
from omegaconf import OmegaConf
from diffusion import logger, dist_util
from diffusion.datasets import load_data, data_loader
from diffusion.script_util import (
    model_and_diffusion_defaults,
    showdata,
    create_model_and_diffusion,
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



def main(args):
    # Configure the logger with specified directory
    logger.configure(dir=args.dir)
    logger.log("**********************************")
    logger.log("log configure")
    # Setup distributed training environment
    dist_util.setup_dist()
    logger.log(f"device information:{dist_util.dev()}")
    # Create and merge configuration files
    basic_conf = create_config()
    input_conf = OmegaConf.load(args.config)
    config = OmegaConf.merge(basic_conf, input_conf)
    
    # load data 
    logger.info(f"The args gene set is: {args.gene_set}")

    train_data = load_data(train_path = config.data.train_path,
                    train_label_path = config.data.train_label_path,
                    gene_selection = config.model.feature_size,
                    class_cond=config.model.class_cond,
                    gene_set = args.gene_set,
                    train=True,
                    random= config.data.random,
                    data_filter=config.data.filter,
    )
    # Save the selected gene features if using random selection
    if config.data.random:
        train_data.save_gene()

    logger.info(f"The size of train dataset: {train_data[:][0].shape}")       
    # Create data loader with batching and potential dropout
    loader = data_loader(train_data,
                    batch_size=config.train.batch_size,
                    drop_fraction = config.data.drop_fraction,            
    )
    # Count batches and samples for logging 
    batch_count = 0
    sample_count = 0
    for batch in loader:
        batch_count += 1
        sample_count += batch[0].shape[0]

    logger.info(f"Total batches: {batch_count}")
    logger.info(f"Total samples: {sample_count}")
    # if dataset[:][0].shape[-1] != config.model.feature_size:
        # config.model.feature_size =  dataset[:][0].shape[-1]
    #     logger.log(f"*{args.gene_set} does not met the gene selection requirement, pick all genes from the set")
    
    # Set model configuration parameters
    config.model.patch_size = config.model.feature_size
    config.model.n_embd = config.model.patch_size * 8
    
    logger.info(config)
    model_config = OmegaConf.to_container(config.model, resolve=True)
    diffusion_config = OmegaConf.to_container(config.diffusion, resolve=True)
    # Create model and diffusion process
    logger.log("creating model and diffusion ... ")
    model, diffusion = create_model_and_diffusion(**model_config, **diffusion_config)
    model.to(dist_util.dev())
    logger.info(model)
    ema = deepcopy(model).to(dist_util.dev())  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    schedule_sampler = create_named_schedule_sampler(config.train.schedule_sampler, diffusion)
    logger.log("train the model:")  
    optimizer = th.optim.Adam(model.parameters(),lr=config.train.lr)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    # Training loop
    for epoch in range(config.train.num_epoch):
        for idx,batch_x in enumerate(loader):
            batch, cond = batch_x
            y = {k: v.to(dist_util.dev()) for k, v in cond.items()}
            batch_size = batch.shape[0]
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            losses = diffusion.loss(model,batch.to(dist_util.dev()),t.to(dist_util.dev()), model_kwargs=y)
            loss = (losses["loss"]).mean()
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            update_ema(ema, model)


        if (epoch % config.train.log_interval == 0):
            logger.log(f"The loss is : {loss} at {epoch} epoch")
        # save model checkpoint     
        if(epoch%config.train.save_interval==0) and epoch > 0:
            if dist.get_rank() == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": optimizer.state_dict(),
                    "conf": config
                }
                filename = f"model{(epoch):05d}.pt"
                checkpoint_path = os.path.join(get_blob_logdir(), filename)
                th.save(checkpoint, checkpoint_path)
                logger.log(f"Saved checkpoint to {checkpoint_path}")
    
    logger.log("training process completed")



def create_config():
    
    defaults = {
        "data":{
            "data_dir": None,
            "dir_out": "results",
            "gene_selection": None,
            "samples":124,
            "drop_fraction":0,
            "random": False
        },
        "train":{
            "microbatch": 16,
            "log_interval": 500,
            "save_interval": 10000,
            "schedule_plot": False,
            "resume_checkpoint": "",
            "ema_rate": 0.9999,
            "num_epoch":10001,
            "schedule_sampler":"uniform",
        },
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




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/random/mrna_16.yaml")
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--gene_set", type=str, default="Random")
    args = parser.parse_args()
    main(args)