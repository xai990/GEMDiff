import os

import numpy as np
import torch as th
from omegaconf import OmegaConf
from diffusion import dist_util, logger
from diffusion.script_util import showdata, find_model, create_model_and_diffusion
import datetime
from diffusion.datasets import load_data


def main():
    logger.configure(dir = 'log/')
    logger.log("**********************************")
    logger.log("log configure")
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    input_conf = OmegaConf.load("configs/mrna_gpt_diffusion.yaml")
    dir_out = os.path.join(input_conf.data.dir_out, now,)
    ckpt_path = input_conf.sample.model_path
    state_dict = find_model(ckpt_path)
    
    config = state_dict["conf"]
    model_config = OmegaConf.to_container(config.model, resolve=True)
    diffusion_config = OmegaConf.to_container(config.diffusion, resolve=True)
    _, diffusion = create_model_and_diffusion(**model_config, **diffusion_config)
    # load data 
    dataset = load_data(data_dir = config.data.data_dir,
                    gene_selection = config.model.feature_size,
                    class_cond=config.model.class_cond,
                    # gene_set = config.data.gene_set,
    )
    logger.log("Plot the original dataset with UMAP")
    showdata(dataset,dir = dir_out, schedule_plot = "origin",)   
    logger.log("Plot the forward diffsuion process with UMAP")
    showdata(dataset,
             dir = dir_out,
             schedule_plot="forward",
             diffusion=diffusion, 
             num_steps = config.diffusion.diffusion_steps,
    )
    # load the fake data 
    file_path = input_conf.plot.file_path
    data_fake = np.load(file_path)
    
    showdata(dataset,dir = dir_out, schedule_plot = "reverse", synthesis_data=data_fake)
    logger.log("plot complete...")


if __name__ == "__main__":
    main()