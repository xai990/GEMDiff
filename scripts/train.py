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
import datetime
import os 
import matplotlib.pyplot as plt


def main(args):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logger.configure(dir=args.dir)
    logger.log("**********************************")
    logger.log("log configure")
    dist_util.setup_dist()
    logger.log(f"device information:{dist_util.dev()}")
    basic_conf = create_config()
    input_conf = OmegaConf.load(args.config)
    config = OmegaConf.merge(basic_conf, input_conf)
    
    # load data 
    dataset = load_data(data_dir = config.data.data_dir,
                    gene_selection = config.model.feature_size,
                    class_cond=config.model.class_cond,
                    gene_set = args.gene_set,
    )
    
    # change the size of the data 
    logger.info(f"The size of dataset: {dataset[:][0].shape}")
    loader = data_loader(dataset,
                    batch_size=config.train.batch_size,            
    )
    
    # configure the dir out 
    dir_out = os.path.join(config.data.dir_out, now,)
    # plot the shape of the data 
    # showdata(dataset,dir = dir_out, schedule_plot = "origin",)     
    config.model.n_embd = config.model.patch_size * 8
   
    logger.info(config)
    model_config = OmegaConf.to_container(config.model, resolve=True)
    diffusion_config = OmegaConf.to_container(config.diffusion, resolve=True)
    
    logger.log("creating model and diffusion ... ")
    model, diffusion = create_model_and_diffusion(**model_config, **diffusion_config)
    model.to(dist_util.dev())
    logger.info(model)
    
    # forward process plot 
    # showdata(dataset,
    #          dir = dir_out,
    #          schedule_plot="forward",
    #          diffusion=diffusion, 
    #          num_steps = config.diffusion.diffusion_steps,
    # )
    
    logger.log("train the model:")  
    optimizer = th.optim.Adam(model.parameters(),lr=config.train.lr)
    for epoch in range(config.train.num_epoch):
        for idx,batch_x in enumerate(loader):
            batch, cond = batch_x
            y = {k: v.to(dist_util.dev()) for k, v in cond.items()}
            batch_size = batch.shape[0]
            t = th.randint(0,config.diffusion.diffusion_steps,size=(batch_size//2,),dtype=th.int32)
            t = th.cat([t,config.diffusion.diffusion_steps-1-t],dim=0)
            losses = diffusion.loss(model,batch.to(dist_util.dev()),t.to(dist_util.dev()), model_kwargs=y)
            loss = (losses["loss"]).mean()
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()

        if (epoch % config.train.log_interval == 0):
            logger.log(f"The loss is : {loss} at {epoch} epoch")
        # save model checkpoint     
        if(epoch%config.train.save_interval==0) and epoch > 0:
            if dist.get_rank() == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "conf": config
                }
                filename = f"model{(epoch):05d}.pt"
                checkpoint_path = os.path.join(get_blob_logdir(), filename)
                th.save(checkpoint, checkpoint_path)
                logger.log(f"Saved checkpoint to {checkpoint_path}")
    
    
    logger.log("completed")



def create_config():
    
    defaults = {
        "data":{
            "data_dir": None,
            "dir_out": "results",
            "gene_selection": None,
        },
        "train":{
            "microbatch": 16,
            "log_interval": 500,
            "save_interval": 8000,
            "schedule_plot": False,
            "resume_checkpoint": "",
            "ema_rate": 0.9999,
            "num_epoch":8001
        }
    }
    defaults.update(model_and_diffusion_defaults())
   
    return defaults  


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mrna_128.yaml")
    parser.add_argument("--dir", type=str, default="log/")
    parser.add_argument("--gene_set", type=str, default=None)
    args = parser.parse_args()
    main(args)