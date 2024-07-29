import os
import argparse
import numpy as np
import torch as th
from omegaconf import OmegaConf
from diffusion import dist_util, logger
from diffusion.script_util import showdata, find_model, create_model_and_diffusion
from diffusion.datasets import load_data


def main(args):
    logger.configure(dir=args.dir)
    logger.log("**********************************")
    logger.log("log configure")
    dist_util.setup_dist()
    logger.log(f"device information:{dist_util.dev()}")
    logger.log("Load the config...")
    basic_conf = create_config()
    input_conf = OmegaConf.load(args.config)
    config = OmegaConf.merge(basic_conf, input_conf)
    # load data 
    train_data, test_data = load_data(data_dir = config.data.data_dir,
                    gene_selection = config.model.feature_size,
                    class_cond=config.model.class_cond,
                    gene_set = args.gene_set,
    )
    # balance the train and test data 
    # train_N, train_T = balance_sample_screen(train_data)
    # test_N, test_T = balance_sample_screen(test_data)
    # logger.debug(f"The shape of trian data is {train_data[:][0].shape}")
    # logger.log("Plot the original dataset with UMAP")
    # showdata(train_data,dir = args.dir_out, schedule_plot = "reverse",)     
    # logger.log("Plot the forward diffsuion process with UMAP")
    # showdata(dataset,
    #          dir = dir_out,
    #          schedule_plot="forward",
    #          diffusion=diffusion, 
    #          num_steps = config.diffusion.diffusion_steps,
    # )
    # load the fake data 
    file_path = args.sample_path
    data_fake = np.load(file_path)
    logger.log("Plot the synthesis data with UMAP")
    showdata(train_data,dir = args.dir_out, schedule_plot = "reverse", synthesis_data=data_fake)
    
    logger.log("plot complete...")



def create_config():
    
    defaults = {
        "data":{
            "data_dir": None,
            "dir_out": "results",
            "gene_selection": None,
            # "samples":124,
        },
        "train":{
            "microbatch": 16,
            "log_interval": 500,
            "save_interval": 8000,
            "schedule_plot": False,
            "resume_checkpoint": "",
            "ema_rate": 0.9999,
            "num_epoch":8001,
            "schedule_sampler":"uniform",
        },
        "umap":{
            "n_neighbors":90,
            "min_dist":0.3,
        }

    }
    # defaults.update(model_and_diffusion_defaults())
   
    return defaults  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/random/mrna_16.yaml")
    parser.add_argument("--dir_out", type=str, default="results/")
    parser.add_argument("--dir", type=str, default="log/")
    parser.add_argument("--sample_path", type=str, default=None)
    parser.add_argument("--gene_set", type=str, default="Random")
    args = parser.parse_args()
    main(args)