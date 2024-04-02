
import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from diffusion.datasets import load_data, datascalar
from diffusion import dist_util, logger
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    find_model,
)
from omegaconf import OmegaConf
import datetime

def main(args):
    #args = create_argparser().parse_args()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    dist_util.setup_dist()
    logger.configure(dir=args.dir)
    # configure the dir out 
    
    logger.log("Load the config...")
    ckpt_path = args.model_path
    state_dict = find_model(ckpt_path)
    basic_conf = create_config()
    loaded_conf = state_dict["conf"]
    config = OmegaConf.merge(basic_conf, loaded_conf)
    dir_out = os.path.join(args.dir_out, now,)
    logger.log(config)
    model_config = OmegaConf.to_container(config.model, resolve=True)
    diffusion_config = OmegaConf.to_container(config.diffusion, resolve=True)
    
    logger.log("Load model ... ")
    model, diffusion = create_model_and_diffusion(**model_config, **diffusion_config)
    model.load_state_dict(state_dict["model"])
    model.to(dist_util.dev())
    model.eval()
    
    logger.log("sampling...")
    all_genes = []
    all_labels = []
    while len(all_genes) * config.sample.batch_size < config.sample.num_samples:
        model_kwargs = {}
        if config.model.class_cond:
            classes = th.randint(
                low=0, high=2, size=(config.sample.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = diffusion.p_sample_loop
        # logger.debug(f"The classes are {classes}")
        sample = sample_fn(
            model,
            (config.sample.batch_size, config.model.feature_size),
            #clip_denoised=config.sample.clip_denoised,
            model_kwargs=model_kwargs,
        )
        logger.info(f"The size of sample is:{sample.size()}")
        
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_genes.extend([sample.cpu().numpy() for sample in gathered_samples])
        if config.model.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_genes) * config.sample.batch_size} samples")

    arr = np.concatenate(all_genes, axis=0)
    arr = arr[: config.sample.num_samples]
    if config.model.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: config.sample.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        os.makedirs(dir_out,exist_ok=True)
        out_path = os.path.join(dir_out, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if config.model.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_config():
    
    defaults = {
        "data":{
            "data_dir": None,
            "dir_out": "results",
            "gene_selection": None,
            "exampledata": False,
        },
        "sample":{
            "batch_size": 128,
            "num_samples": 800,
            "model_path": 250,
            "clip_denoised": False,
        }
    }
    
    defaults.update(model_and_diffusion_defaults())
   
    return defaults  






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="log/mrna_8/2024-03-28-13-18/model08000.pt")
    parser.add_argument("--dir_out", type=str, default="results/")
    parser.add_argument("--dir", type=str, default="log/")
    args = parser.parse_args()
    main(args)