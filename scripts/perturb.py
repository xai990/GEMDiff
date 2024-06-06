import argparse
from omegaconf import OmegaConf
from diffusion import logger, dist_util
from diffusion.datasets import load_data, data_loader, balance_sample_screen,LabelGeneDataset
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
import datetime

def main(args):
    #args = create_argparser().parse_args()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
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
    train_N, train_T = balance_sample_screen(train_data)
    # test_N, test_T = balance_sample_screen(test_data)
    # set the model param
    
    logger.log("creating model and diffusion ... ")
    config.model.feature_size = train_N.shape[-1]
    logger.info(f"The model feature size is : {config.model.feature_size}")
    config.model.patch_size = config.model.feature_size
    # config.model.n_embd = config.model.patch_size * 8
    config.model.n_embd = config.model.patch_size * 2
    # config.model.class_cond = False # not specific the label during the training in perturbation task
    logger.info(config)
    model_config = OmegaConf.to_container(config.model, resolve=True)
    diffusion_config = OmegaConf.to_container(config.diffusion, resolve=True)
    # model, diffusion = create_model_and_diffusion(**model_config, **diffusion_config)
    # model.to(dist_util.dev())
    # logger.info(model)
    # ema = deepcopy(model).to(dist_util.dev())  # Create an EMA of the model for use after training
    # requires_grad(ema, False)
    if args.model_dir is None:
        model_N, diffusion = create_model_and_diffusion(**model_config, **diffusion_config)
        model_N.to(dist_util.dev())
        logger.info(model_N)
        ema_N = deepcopy(model_N).to(dist_util.dev())  # Create an EMA of the model for use after training
        requires_grad(ema_N, False)
        logger.log("train the target model:")  
        
        schedule_sampler = create_named_schedule_sampler(config.train.schedule_sampler, diffusion)
        train_N = LabelGeneDataset(train_N,0)
        loader_N = data_loader(train_N,
                        batch_size=config.train.batch_size,            
        ) 
        optimizer_N = th.optim.Adam(model_N.parameters(),lr=config.train.lr)
        update_ema(ema_N, model_N, decay=0)  # Ensure EMA is initialized with synced weights
        model_N.train()  # important! This enables embedding dropout for classifier-free guidance
        ema_N.eval()  # EMA model should always be in eval mode
        for epoch in range(config.train.num_epoch):
            for idx,batch_x in enumerate(loader_N):
                batch, cond = batch_x
                target_y = {k: v.to(dist_util.dev()) for k, v in cond.items()}
                batch_size = batch.shape[0]
                # t = th.randint(0,config.diffusion.diffusion_steps,size=(batch_size//2,),dtype=th.int32)
                # t = th.cat([t,config.diffusion.diffusion_steps-1-t],dim=0)
                t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
                losses = diffusion.loss(model_N,batch.to(dist_util.dev()),t.to(dist_util.dev()), model_kwargs=target_y)
                loss = (losses["loss"]).mean()
                optimizer_N.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(model_N.parameters(),1.)
                optimizer_N.step()
                update_ema(ema_N, model_N)


            if (epoch % config.train.log_interval == 0):
                logger.log(f"The loss is : {loss} at {epoch} epoch")
            # save model checkpoint     
            if(epoch%config.train.save_interval==0) and epoch > 0:
                if dist.get_rank() == 0:
                    checkpoint = {
                        "model": model_N.state_dict(),
                        "ema": ema_N.state_dict(),
                        "opt": optimizer_N.state_dict(),
                        "conf": config
                    }
                    filename = f"model_N{(epoch):05d}.pt"
                    checkpoint_path = os.path.join(get_blob_logdir(), filename)
                    th.save(checkpoint, checkpoint_path)
                    logger.log(f"Saved checkpoint to {checkpoint_path}")
        
        logger.log("train the source model:")  
        model_T, _ = create_model_and_diffusion(**model_config, **diffusion_config)
        model_T.to(dist_util.dev())
        logger.info(model_T)
        ema_T = deepcopy(model_T).to(dist_util.dev())  # Create an EMA of the model for use after training
        requires_grad(ema_T, False)
        train_T = LabelGeneDataset(train_T,1)
        loader_T = data_loader(train_T,
                        batch_size=config.train.batch_size,            
        )
        optimizer_T = th.optim.Adam(model_T.parameters(),lr=config.train.lr)
        update_ema(ema_T, model_T, decay=0)  # Ensure EMA is initialized with synced weights
        model_T.train()  # important! This enables embedding dropout for classifier-free guidance
        ema_T.eval()  # EMA model should always be in eval mode
        for epoch in range(config.train.num_epoch):
            for idx,batch_x in enumerate(loader_T):
                batch, cond = batch_x # not consider the label info for now 
                source_y = {k: v.to(dist_util.dev()) for k, v in cond.items()}
                batch_size = batch.shape[0]
                t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
                losses = diffusion.loss(model_T,batch.to(dist_util.dev()),t.to(dist_util.dev()), model_kwargs=y)
                loss = (losses["loss"]).mean()
                optimizer_N.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(model_T.parameters(),1.)
                optimizer_T.step()
                update_ema(ema_T, model_T)


            if (epoch % config.train.log_interval == 0):
                logger.log(f"The loss is : {loss} at {epoch} epoch")
            # save model checkpoint     
            if(epoch%config.train.save_interval==0) and epoch > 0:
                if dist.get_rank() == 0:
                    checkpoint = {
                        "model": model_T.state_dict(),
                        "ema": ema_T.state_dict(),
                        "opt": optimizer_T.state_dict(),
                        "conf": config
                    }
                    filename = f"model_T{(epoch):05d}.pt"
                    checkpoint_path = os.path.join(get_blob_logdir(), filename)
                    th.save(checkpoint, checkpoint_path)
                    logger.log(f"Saved checkpoint to {checkpoint_path}")
    else:
        # load the model from pretrain
        model_N, diffusion = create_model_and_diffusion(**model_config, **diffusion_config)
        ema_N = deepcopy(model_N)
        model_T, _ = create_model_and_diffusion(**model_config, **diffusion_config)
        ema_T = deepcopy(model_T)
        state_dict = th.load(f"{args.model_dir}/model_N40000.pt")
        ema_N.load_state_dict(state_dict["ema"])
        ema_N.to(dist_util.dev())
        ema_N.eval()
        state_dict = th.load(f"{args.model_dir}/model_T40000.pt")
        ema_T.load_state_dict(state_dict["ema"])
        ema_T.to(dist_util.dev())
        ema_T.eval()
    
    logger.log("pertubing the source to target")
    source_label = np.array(test_T.shape[0] * [1] if args.vaild else train_T.shape[0] * [1])
    noise_T = diffusion.ddim_reverse_sample_loop(
        ema_T,
        test_T.shape if args.vaild else train_T.shape,
        th.Tensor(test_T).float().to(dist_util.dev()) if args.vaild else th.Tensor(train_T).float().to(dist_util.dev()),
        clip_denoised=False,
        model_kwargs=source_label,
        # device=dist_util.dev(),
    )
    target_label = np.array(noise_T.shape[0] * [0])
    target = diffusion.ddim_sample_loop(ema_N,noise_T.shape,noise= noise_T,clip_denoised=False,model_kwargs=target_label)
    shape_str = "x".join([str(x) for x in noise_T.shape])
    out_path = os.path.join(get_blob_logdir(), f"reverse_sample_{shape_str}.npz")
    np.savez(out_path, target.cpu().numpy())
    logger.log(f"saving perturb data array to {out_path}")
    logger.log("visulize the perturbed data and real data")
    plotdata = [test_N, test_T, target] if args.vaild else [train_N, train_T, target]
    showdata(plotdata,dir = get_blob_logdir(), schedule_plot = "perturb", n_neighbors =config.umap.n_neighbors,min_dist=config.umap.min_dist)
    
    logger.log("filter the perturbed gene -- 1 std")
    gene_index = filter_gene(train_T, target.cpu().numpy())
    logger.log(f"The indentified genes are: {train_data.find_gene(gene_index)} -- 1 standard deviation of the perturbation among all {train_N.shape[1]} gene")
    logger.log("pertubing complete")



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
            "save_interval": 10000,
            "schedule_plot": False,
            "resume_checkpoint": "",
            "ema_rate": 0.9999,
            "num_epoch":40001,
            "schedule_sampler":"uniform",
        },
        "perturb":{
            # "samples":124,
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
    parser.add_argument("--config", type=str, default="configs/mrna_16.yaml")
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--gene_set", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
