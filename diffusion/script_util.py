import argparse 
import os 
import matplotlib.pyplot as plt
from . import gaussian_diffusion as gd 
from .respace import SpacedDiffusion, space_timesteps
import torch as th 
from .mlp import GPT , Classifier
from . import logger 
import umap.plot
import numpy as np 
import plotly.graph_objects as go
import random
from sklearn.metrics import silhouette_score 
from .datasets import sample_screen, balance_sample


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
            "diffusion_steps": 4000,
            "noise_schedule": "linear",
            "linear_start": 0.0015,
            "linear_end": 0.0195,
            "log_every_t": 10,
            "schedule_sampler": "uniform",
            "learn_sigma": True,
            "rescale_timesteps": True, 
            "timestep_respacing":"",
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
    learn_sigma,
    rescale_timesteps,
    timestep_respacing,
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
        learn_sigma = learn_sigma,
    )
    diffusion = create_diffusion(
        steps= diffusion_steps,
        noise_schedule = noise_schedule,
        linear_start= linear_start,
        linear_end = linear_end,
        log_every_t = log_every_t,
        learn_sigma = learn_sigma,
        rescale_timesteps = rescale_timesteps,
        timestep_respacing=timestep_respacing,
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
        learn_sigma = True,
    ):
    
    return GPT(
        gene_feature=feature_size,
        n_embd=n_embd,
        n_head=n_head,
        dropout=dropout,
        n_layer = n_layer,
        patch_size =patch_size,
        num_classes=(NUM_CLASSES if class_cond else None),
        learn_sigma = learn_sigma,
    )

def create_diffusion(
    *,
    steps = 1000,
    noise_schedule = "linear",
    linear_start = 0.0015,
    linear_end = 0.0195,
    log_every_t = 10,
    learn_sigma = True, 
    rescale_timesteps = False,
    timestep_respacing = "",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps, linear_start,linear_end)
    # logger.debug(f"The betas is {betas} -- script")
    loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [steps]
    # return DenoiseDiffusion(
    #     betas = betas,
    #     log_every_t = log_every_t,
    #     loss_type = loss_type,
    #     model_mean_type = gd.ModelMeanType.EPSILON,
    #     model_var_type = gd.ModelVarType.LEARNED if learn_sigma else gd.ModelVarType.FIXED
    # )
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.LEARNED if learn_sigma else gd.ModelVarType.FIXED,
        log_every_t = log_every_t,
        loss_type = loss_type,
        rescale_timesteps = rescale_timesteps,
    )


def create_classifier(
        *,
        feature_size,
        class_cond,
        n_head,
        n_layer = 4,
        dropout = 0.0,
        learn_sigma = True,
    ):
    
    return Classifier(
        gene_feature=feature_size,
        n_head=n_head,
        dropout=dropout,
        n_layer = n_layer,
        num_classes=(NUM_CLASSES if class_cond else None),
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
            synthesis_data = None,
            n_neighbors = 15,
            min_dist = 0.1,
            random_state = 41,
            gene_set = None,
    ):
    # dir to save the plot image
    assert isinstance(dir,str)
    os.makedirs(dir,exist_ok=True)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    added_labels = set()
    geneset = gene_set.split("/")[-1] if gene_set else "random"
    if schedule_plot == "forward": 
        colors = ['blue','orange']
        labels = ['normal','tumor']
        assert diffusion is not None
        rows = num_shows // cols
        fig,axes = plt.subplots(rows,cols,figsize=(28,3))
        plt.rc('text',color='black')
        data , y = dataset[:][0], dataset[:][1]['y']
        data = th.from_numpy(data).float()
        for i in range(num_shows):
            j = i // cols
            k = i % cols
            t = th.full((data.shape[0],),i*num_steps//num_shows)
            x_i = diffusion.q_sample(data,t)
            q_i = reducer.fit_transform(x_i)
            added_labels.clear()
            for ele in y:
                index_mask = (ele==y)
                if np.any(index_mask):
                    label = labels[ele] if ele not in added_labels else None
                    axes[j,k].scatter(q_i[index_mask,0],
                                q_i[index_mask,1],
                                label = label, 
                                color=colors[ele],
                                edgecolor='white')
                    if label:
                        added_labels.add(ele)
            axes[j,k].set_axis_off()
            axes[j,k].set_title('$q(\mathbf{x}_{'+str(i*num_steps//num_shows)+'})$')
        handles, labels = axes[0, 0].get_legend_handles_labels()  # Get handles and labels from any one subplot
        fig.legend(handles, labels, loc='upper left', ncol=3)
        plt.savefig(f"{dir}/dataset-forward_{data.shape[-1]}.png")
        plt.close()
    elif schedule_plot == "origin":
        # fig,ax = plt.subplots()      
        data , y = dataset[:][0], dataset[:][1]['y']
        # Define the range of parameters you want to explore
        # n_neighbors_options = [15, 30, 60, 90, 120, 150, 180]
        # min_dist_options = [0.1, 0.3, 0.6, 0.9]
        color_map = ['blue','orange']
        fig,ax = plt.subplots()
        labels = ['normal','tumor']
        embedding = reducer.fit_transform(data)
        for ele in y:
            index_mask = (ele==y)
            if np.any(index_mask):
                label = labels[ele] if labels[ele] not in added_labels else None
                ax.scatter(embedding[index_mask,0],
                            embedding[index_mask,1],
                            label = label, 
                            color=color_map[ele],
                            edgecolor='white')
                if label:
                    added_labels.add(label)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        # plt.legend(loc="upper left")
        ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.savefig(f"{dir}/dataset_{data.shape[-1]}.png")
        plt.close()
    elif schedule_plot == "balance":
        # logger.debug(f"The dataset is : {dataset[:][1]} -- script_util") 
        color_map = ['blue','orange']
        fig,axs = plt.subplots()
        labels = ['normal','tumor']
        dataset_N,dataset_T = sample_screen(dataset)
        dataset_N, dataset_T = balance_sample([dataset_N,dataset_T])
        data_merge = np.vstack([dataset_N,dataset_T])
        x_ump = reducer.fit_transform(data_merge)
        q_i = x_ump[:len(dataset_N)]
        q_x = x_ump[len(dataset_N):]
        # use Silhouette Score as standard for the plot 
        # logger.info("*********************************************************")
        # logger.log(f"{geneset} experiemnt of silhouette score: {score}")
        # logger.info("*********************************************************")
        # titles = [f"Geneset {geneset} with silhouette score {score:.3f}"]
        axs.scatter(q_i[:,0],q_i[:,1],color = color_map[0],edgecolor='white',label=labels[0])
        axs.scatter(q_x[:,0],q_x[:,1],color = color_map[1],edgecolor='white',label=labels[1])
        # axs.set_title(titles[0])
        axs.axis('off')
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
        # ax.axes.xaxis.set_ticklabels([])
        # ax.axes.yaxis.set_ticklabels([])
        plt.tight_layout(pad=1.0)
        plt.savefig(f"{dir}/{geneset}_{data_merge.shape[-1]}.png")
        plt.close()
        """
        # Setup the subplot grid
        fig, axes = plt.subplots(nrows=len(n_neighbors_options), ncols=len(min_dist_options), figsize=(15, 15))
        for i, n_neighbors in enumerate(n_neighbors_options):
            for j, min_dist in enumerate(min_dist_options):
                # Create UMAP model
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=41)
                embedding = reducer.fit_transform(data)
                # Plot
                ax = axes[i,j]  # Get a specific subplot axis
                added_labels.clear()
                for ele in y:
                    index_mask = (ele==y)
                    if np.any(index_mask):
                        label = labels[ele] if labels[ele] not in added_labels else None
                        ax.scatter(embedding[index_mask,0],
                                    embedding[index_mask,1],
                                    label = label, 
                                    color=color_map[ele],
                                    edgecolor='white')
                        if label:
                            added_labels.add(label)
                # scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='viridis', s=5)
                # ax.set_title(f'n_neighbors={n_neighbors}, min_dist={min_dist}')
                if i == 0:
                    ax.set_xlabel(f'{min_dist}')
                    ax.xaxis.set_label_position('top')
                if j == 0:
                    ax.set_ylabel(f'{n_neighbors}')
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                # ax.axis('off')
        """
        # handles, labels = axes[0, 0].get_legend_handles_labels()
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper left', ncol=3)
        # fig.text(0.5, 1,'min_dist',ha='center',)
        # fig.text(0.0, 0.5,'n_neighbors', va='center', rotation='vertical')
       
    elif schedule_plot == "reverse":
        data_r , y_r = dataset[:][0], dataset[:][1]['y']
        data_f, y_f = synthesis_data["arr_0"], synthesis_data["arr_1"]
        # stack the data together for overarching patterns 
        data_merged = np.vstack([data_r,data_f])
        fig,ax = plt.subplots()
        color_map = ['blue','orange','cyan','blueviolet']
        labels = ['real_normal','real_tumor','fake_normal', 'fake_turmor']
        
        q_i = reducer.fit_transform(data_merged)
        q_r = q_i[:len(y_r)]
        q_f = q_i[len(y_r):]
        # Plot
        for ele in y_r:
            index_mask = (ele==y_r)
            if np.any(index_mask):
                label = labels[ele] if labels[ele] not in added_labels else None
                ax.scatter(q_r[index_mask,0],
                            q_r[index_mask,1],
                            label = label, 
                            color=color_map[ele],
                            edgecolor='white')
                if label:
                    added_labels.add(label)
        for ele in y_f:
            index_mask = (ele==y_f)
            if np.any(index_mask):
                label = labels[ele+2] if labels[ele+2] not in added_labels else None
                ax.scatter(q_f[index_mask,0],
                            q_f[index_mask,1],
                            label = label, 
                            color=color_map[ele+2],
                            edgecolor='white',
                            alpha=0.5)
                if label:
                    added_labels.add(label)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        """
        # Setup the subplot grid
        fig, axes = plt.subplots(nrows=len(n_neighbors_options), ncols=len(min_dist_options), figsize=(15, 15))
        for i, n_neighbors in enumerate(n_neighbors_options):
            for j, min_dist in enumerate(min_dist_options):
                # Create UMAP model
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=41)
                q_i = reducer.fit_transform(data_merged)
                q_r = q_i[:len(y_r)]
                q_f = q_i[len(y_r):]
                # Plot
                ax = axes[i, j]  # Get a specific subplot axis
                added_labels.clear()
                for ele in y_r:
                    index_mask = (ele==y_r)
                    if np.any(index_mask):
                        label = labels[ele] if labels[ele] not in added_labels else None
                        ax.scatter(q_r[index_mask,0],
                                    q_r[index_mask,1],
                                    label = label, 
                                    color=color_map[ele],
                                    edgecolor='white')
                        if label:
                            added_labels.add(label)

                for ele in y_f:
                    index_mask = (ele==y_f)
                    if np.any(index_mask):
                        label = labels[ele+2] if labels[ele+2] not in added_labels else None
                        ax.scatter(q_f[index_mask,0],
                                    q_f[index_mask,1],
                                    label = label, 
                                    color=color_map[ele+2],
                                    edgecolor='white')
                        if label:
                            added_labels.add(label)

                # scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='viridis', s=5)
                # ax.set_title(f'n_neighbors={n_neighbors}, min_dist={min_dist}')
                if i == 0:
                    ax.set_xlabel(f'{min_dist}')
                    ax.xaxis.set_label_position('top')
                if j == 0:
                    ax.set_ylabel(f'{n_neighbors}')
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                # ax.axis('off')
        """
        # handles, labels = axes[0, 0].get_legend_handles_labels()
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper left', ncol=3)
        # fig.text(0.5, 1,'min_dist',ha='center',)
        # fig.text(0.0, 0.5,'n_neighbors', va='center', rotation='vertical')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0., hspace=0.)
        # plt.show()
        plt.legend(loc="upper right")
        plt.savefig(f"{dir}/UMAP_plot_realvsfake_{data_r.shape[-1]}.png")
        plt.close()
        """
        # Create a Plotly figure
        fig = go.Figure()
        # plot the synthesis data
        for ele in np.unique(y_f):
            index_mask = (ele==y_f)
            if np.any(index_mask):
                # Add the first scatter plot to the figure
                label = f'Synthesis {ele}'
                fig.add_trace(go.Scatter(x=q_f[index_mask,0], y=q_f[index_mask,1],
                                        mode='markers', 
                                        name=label,textposition='top center')
                                        )

        # Add the original data scatter plot to the figure
        for ele in np.unique(y_r):
            index_mask = (ele==y_r)
            if np.any(index_mask):
                label = f'Real {ele}'
                fig.add_trace(go.Scatter(x=q_r[index_mask,0], y=q_r[index_mask,1],
                                        mode='markers', 
                                        name=label,textposition='top center')
                                        )
                        

        # Update the layout
        fig.update_layout(title='UMAP Plots with Real mRNA data and synthetic mRNA data',
                        xaxis_title='X Axis',
                        yaxis_title='Y Axis',
                        legend_title='Datasets')

        # Show the figure
        fig.write_html(f"{dir}/UMAP_plot_realvsfake_{data_r.shape[-1]}.html")
        """
    elif schedule_plot == "perturb":
        dataset_N, dataset_T, target = dataset 
        x_merged = np.vstack([dataset_N,dataset_T, target])
        x_ump = reducer.fit_transform(x_merged)
        q_n = x_ump[:len(dataset_N)]
        q_t = x_ump[len(dataset_N):(len(dataset_N)+len(dataset_T))]
        q_x = x_ump[(len(dataset_N)+len(dataset_T)):]
        fig,ax = plt.subplots()      
        ax.scatter(q_n[:,0],q_n[:,1],color = 'blue',edgecolor='white',label="real normal")
        ax.scatter(q_t[:,0],q_t[:,1],color = 'orange',edgecolor='white',label="real tumor")
        ax.scatter(q_x[:,0],q_x[:,1],color = 'cyan',edgecolor='white',label="perturb tumor")
        plt.legend(loc="upper right")
        # plt.title("Perturb tumor mRNA expression back to normal with Diffusion model")
        # ax.axis('off')
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.savefig(f"{dir}/UMAP_plot_perturb_{dataset_N.shape[-1]}.png")
        plt.close()
    else:
        raise NotImplementedError(f"unknown schedule plot:{schedule_plot}")



def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find model checkpoint at {model_name}'
    checkpoint = th.load(model_name)
    return checkpoint 



def filter_gene(real, perturb):
    assert real.shape == perturb.shape, f'The datashape of real and perturbed are different'
    # Calculate the difference between the corresponding gene
    logger.log(f"The real data mean {real.mean(axis=0)}-- script_util")
    logger.log(f"The real data std {real.std(axis=0)}-- script_util")
    logger.log(f"The perturb data mean {perturb.mean(axis=0)}-- script_util")
    logger.log(f"The perturb data std {perturb.std(axis=0)}-- script_util")
    differences = np.abs(real - perturb).mean(axis=0)
    logger.log(f"The differences between real and perturb data {differences} -- script_util")
    
    # Calculate the standard deviation of the differences for each gene 
    std_deviation = np.std(differences)
    logger.log(f"The standard deviation between real and perturb data data {std_deviation} -- script_util")
    # fileter columns where the difference is greater than 1 standard deviation 
    perturb_mean = differences.mean()
    logger.log(f"The mean between real and perturb data data {perturb_mean} -- script_util")
    # filter_genecoln = [( differences > perturb_mean + std_deviation) | (differences < perturb_mean - std_deviation)]
    index_array = np.arange(0,real.shape[1])
    filter_index = differences > perturb_mean + std_deviation
    perturb_perc = np.divide(differences, real.mean(axis=0))
    logger.log(f"The real data {real.mean(axis=0)[filter_index]}-- script_util")
    logger.log(f"The perturb data {perturb.mean(axis=0)[filter_index]}-- script_util")
    logger.log(f"The perturbation percentages between real and perturb data data {perturb_perc[filter_index]}-- script_util")
    return index_array[filter_index]



def get_silhouettescore(
    dataset,
    embed_q1 = None,
    embed_q2 = None,  
    n_neighbors = 15,
    min_dist = 0.1,
    random_state = 41,
    gene_set = None,
    balance=False,
):
    
    if embed_q1 is None or embed_q2 is None:
        dataset_N, dataset_T =  sample_screen(dataset)
        if balance:
            dataset_N, dataset_T = balance_sample([dataset_N, dataset_T])
        data_merge = np.vstack([dataset_N,dataset_T])
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        umap_embed = reducer.fit_transform(data_merge)
        embed_q1 = umap_embed[:len(dataset_N)]
        embed_q2 = umap_embed[len(dataset_N):]
    q_i = embed_q1
    q_x = embed_q2
    # use Silhouette Score as standard for the plot 
    x_silhoutette = np.vstack((q_i, q_x))
    # logger.debug(f"The size of x_silhoutette is: {x_silhoutette.shape} -- script_util")
    label_silhoutette = np.vstack((np.zeros((len(embed_q1),1)), np.ones((len(embed_q2),1))))
    score = silhouette_score(x_silhoutette, label_silhoutette)
    return score 



