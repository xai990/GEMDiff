import argparse 
import os 
import datetime 
import matplotlib.pyplot as plt
from . import gaussian_diffusion as gd 
from .gaussian_diffusion import DenoiseDiffusion
import torch as th 
from .mlp import GPT
from . import logger 
import umap.plot
import numpy as np 

NUM_CLASSES = 2

def model_and_diffusion_defaults():
    """
    Defaults for model and diffusion model
    """
    return {
        "model":{
            "feature_size": 2,
            "patch_size": 2,
            "dropout": 0.0,
            "class_cond": False,
            "n_embd": 768,
            "n_head": 4,
            "n_layer": 4,
        },
        "diffusion":{
            "diffusion_steps": 1000,
            "noise_schedule": "linear",
            "linear_start": 0.0015,
            "linear_end": 0.0195,
            "log_every_t": 10,
            "schedule_sampler": "uniform",
        }
    }


def create_model_and_diffusion(
    *,
    feature_size,
    class_cond,
    n_embd,
    n_head,
    dropout,
    diffusion_steps,
    noise_schedule,
    linear_start,
    linear_end,
    log_every_t,
    n_layer,
    patch_size,
    **kwargs,
):
    model = create_model(
        feature_size = feature_size,
        n_embd = n_embd,
        n_head = n_head,
        dropout = dropout,
        class_cond = class_cond,
        n_layer = n_layer,
        patch_size = patch_size,
    )
    diffusion = create_diffusion(
        steps= diffusion_steps,
        noise_schedule = noise_schedule,
        linear_start= linear_start,
        linear_end = linear_end,
        log_every_t = log_every_t,
    )
    return model, diffusion


def create_model(
        *,
        feature_size,
        n_embd,
        class_cond,
        patch_size,
        n_head,
        n_layer = 4,
        dropout = 0.0,
        
    ):
    
    return GPT(
        gene_feature=feature_size,
        n_embd=n_embd,
        n_head=n_head,
        dropout=dropout,
        n_layer = n_layer,
        patch_size =patch_size,
        num_classes=(NUM_CLASSES if class_cond else None),
    )

def create_diffusion(
    *,
    steps = 1000,
    noise_schedule = "linear",
    linear_start = 0.0015,
    linear_end = 0.0195,
    log_every_t = 10,

):
    betas = gd.get_named_beta_schedule(noise_schedule, steps, linear_start,linear_end)
    # logger.debug(f"The betas is {betas} -- script")
    return DenoiseDiffusion(
        betas = betas,
        log_every_t = log_every_t,

    )


def zero_grad(model_params):
    for param in model_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()



def showdata(dataset, 
            dir="results", 
            schedule_plot = "forward", 
            diffusion=None,
            num_steps=1000,
            num_shows=20,
            cols = 10,
            # epoch = 1,
            exampledata = False,
            final_data = False,
            scaler = None,
    ):
    # dir to save the plot image
    assert isinstance(dir,str)
    os.makedirs(dir,exist_ok=True)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    colors = ['blue','orange']
    if schedule_plot == "forward": 
        assert diffusion is not None
        rows = num_shows // cols
        fig,axs = plt.subplots(rows,cols,figsize=(28,3))
        plt.rc('text',color='black')
        if exampledata == True:
            data = dataset[:]
            for i in range(num_shows):
                j = i//10
                k = i%10
                t = th.full((data.shape[0],),i*num_steps//num_shows)
                # logger.debug(f"The t is: {t} -- script_util")
                q_i = diffusion.q_sample(data, t)
                axs[j,k].scatter(q_i[:,0].cpu(),q_i[:,1].cpu(),color='red',edgecolor='white')
                axs[j,k].set_axis_off()
                axs[j,k].set_title('$q(\mathbf{x}_{'+str(i*num_steps//num_shows)+'})$')
        else:
            data , y = dataset[:][0], dataset[:][1]['y']
            data = th.from_numpy(data).float()
            for i in range(num_shows):
                j = i // cols
                k = i % cols
                t = th.full((data.shape[0],),i*num_steps//num_shows)
                x_i = diffusion.q_sample(data,t)
                q_i = reducer.fit_transform(x_i)
                for ele in y:
                    index_mask = (ele==y)
                    if np.any(index_mask):
                        axs[j,k].scatter(q_i[index_mask,0],
                                    q_i[index_mask,1],
                                    label = ele, 
                                    color=colors[ele],
                                    edgecolor='white')
                axs[j,k].set_axis_off()
                axs[j,k].set_title('$q(\mathbf{x}_{'+str(i*num_steps//num_shows)+'})$')
        plt.savefig(f"{dir}/dataset-forward.png")
        plt.close()
    elif schedule_plot == "origin":
        fig,ax = plt.subplots()      
        if exampledata == True:
            data = dataset[:].T.cpu()
            ax.scatter(*data, color = 'blue', edgecolor = 'white')
        else:
            data , y = dataset[:][0], dataset[:][1]['y']
            q_i = reducer.fit_transform(data)
            for ele in y:
                index_mask = (ele==y)
                if np.any(index_mask):
                    ax.scatter(q_i[index_mask,0],
                                q_i[index_mask,1],
                                label = ele, 
                                color=colors[ele],
                                edgecolor='white')
        # ax.scatter(q_i[:,0], q_i[:,1],color='red',edgecolor='white')
        ax.axis('off')
        plt.savefig(f"{dir}/dataset.png")
        plt.close()
    elif schedule_plot == "reverse":
        if exampledata == True:
            fig,axs = plt.subplots(1,10,figsize=(28,3))
            for i in range(1,11):
                cur_x = dataset[i*10].detach().cpu()
                axs[i-1].scatter(cur_x[:,0],cur_x[:,1],color='red',edgecolor='white');
                axs[i-1].set_axis_off();
                axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$')
        else:
            if final_data:
                fig,ax = plt.subplots()
                data = dataset['arr_0'].reshape(400,-1)
                mins, maxs = scaler[0], scaler[1] 
                data = (data+ mins) * (maxs - mins)
                y = dataset['arr_1']
                q_i = reducer.fit_transform(data)
                for ele in y:
                    index_mask = (ele==y)
                    if np.any(index_mask):
                        ax.scatter(q_i[index_mask,0],
                                    q_i[index_mask,1],
                                    label = ele, 
                                    color=colors[ele],
                                    edgecolor='white')
                ax.axis('off')
            else:
                fig,axs = plt.subplots(1,10,figsize=(28,3))
                B, *feature = dataset.shape
                dataset = dataset.reshape(B, -1)
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
                for i in range(1,11):
                    pi_x = dataset[i*10].detach().cpu()
                    pi = reducer.fit_transform(pi_x)
                    axs[i-1].scatter(pi[:,0],pi[:,1],color='orange',edgecolor='white');
                    axs[i-1].set_axis_off();
                    axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$')
        plt.savefig(f"{dir}/dataset-reversed.png")
        plt.close()
    else:
        raise NotImplementedError(f"unknown schedule plot:{schedule_plot}")