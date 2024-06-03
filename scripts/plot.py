import os
import argparse
import numpy as np
import torch as th
from omegaconf import OmegaConf
from diffusion import dist_util, logger
from diffusion.script_util import showdata, find_model, create_model_and_diffusion
import datetime
from diffusion.datasets import load_data


def main(args):
    logger.configure(dir=args.dir)
    logger.log("**********************************")
    logger.log("log configure")
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    dist_util.setup_dist()
    logger.log(f"device information:{dist_util.dev()}")
    ckpt_path = args.model_path
    dir_out = os.path.join(args.dir_out, now,)
    state_dict = find_model(ckpt_path)
    
    config = state_dict["conf"]
    config.diffusion.learn_sigma = False
    model_config = OmegaConf.to_container(config.model, resolve=True)
    diffusion_config = OmegaConf.to_container(config.diffusion, resolve=True)
    # _, diffusion = create_model_and_diffusion(**model_config, **diffusion_config)
    # load data 
    _, dataset = load_data(data_dir = config.data.data_dir,
                    gene_selection = config.model.feature_size,
                    class_cond=config.model.class_cond,
                    gene_set = args.gene_set,
    )
    
    logger.log("Plot the original dataset with UMAP")
    showdata(dataset,dir = dir_out, schedule_plot = "origin",)   
    logger.log("Plot the forward diffsuion process with UMAP")
    # showdata(dataset,
    #          dir = dir_out,
    #          schedule_plot="forward",
    #          diffusion=diffusion, 
    #          num_steps = config.diffusion.diffusion_steps,
    # )
    # load the fake data 
    file_path = args.sample_path
    data_fake = np.load(file_path)
    
    showdata(dataset,dir = dir_out, schedule_plot = "reverse", synthesis_data=data_fake)
    logger.log("plot complete...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="log/mrna_8/2024-03-28-13-18/model08000.pt")
    parser.add_argument("--dir_out", type=str, default="results/")
    parser.add_argument("--dir", type=str, default="log/")
    parser.add_argument("--sample_path", type=str, default="results/2024-03-27T13-50-34/samples_400x8.npz")
    parser.add_argument("--gene_set", type=str, default=None)
    args = parser.parse_args()
    main(args)