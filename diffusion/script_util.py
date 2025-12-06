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
# import plotly.graph_objects as go
import random
from sklearn.metrics import silhouette_score 
from .datasets import sample_screen, balance_sample
import matplotlib.cm as cm

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
            schedule_plot="forward", 
            diffusion=None,
            num_steps=1000,
            num_shows=20,
            cols=10,
            synthesis_data=None,
            n_neighbors=15,
            min_dist=0.1,
            random_state=41,
            gene_set=None,
            class_names=None  # NEW: Pass your class names here e.g., ['Normal', 'Tumor']
    ):
    # dir to save the plot image
    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    added_labels = set()
    geneset = gene_set.split("/")[-1] if gene_set else "random"

    # Default labels if not provided
    if class_names is None:
        labels_map = ['normal', 'tumor']
    else:
        labels_map = class_names

    if schedule_plot == "forward": 
        # Use generic color map to handle any number of classes
        colors = cm.get_cmap('tab10').colors
        assert diffusion is not None
        rows = num_shows // cols
        fig, axes = plt.subplots(rows, cols, figsize=(28, 3))
        plt.rc('text', color='black')
        data, y = dataset[:][0], dataset[:][1]['y']
        data = th.from_numpy(data).float()
        for i in range(num_shows):
            j = i // cols
            k = i % cols
            t = th.full((data.shape[0],), i * num_steps // num_shows)
            x_i = diffusion.q_sample(data, t)
            q_i = reducer.fit_transform(x_i)
            added_labels.clear()
            for ele in y:
                index_mask = (ele == y)
                if np.any(index_mask):
                    # Robust label lookup
                    label_text = labels_map[ele] if ele < len(labels_map) else f"Class {ele}"
                    label = label_text if ele not in added_labels else None
                    
                    axes[j, k].scatter(q_i[index_mask, 0],
                                q_i[index_mask, 1],
                                label=label, 
                                color=colors[ele % len(colors)],
                                edgecolor='white')
                    if label:
                        added_labels.add(ele)
            axes[j, k].set_axis_off()
            axes[j, k].set_title('$q(\mathbf{x}_{'+str(i*num_steps//num_shows)+'})$')
        handles, labels = axes[0, 0].get_legend_handles_labels() 
        fig.legend(handles, labels, loc='upper left', ncol=3)
        plt.savefig(f"{dir}/dataset-forward_{data.shape[-1]}.png")
        plt.close()

    elif schedule_plot == "origin":
        train_data, test_data = dataset[0], dataset[1]
        train, train_y = train_data[:][0], train_data[:][1]['y']
        test, test_y = test_data[:][0], test_data[:][1]['y']
        data_merge = np.vstack([train, test])
        
        color_map = ['blue', 'orange']
        fig, ax = plt.subplots()
        labels = ['train', 'test']
        x_ump = reducer.fit_transform(data_merge)
        q_i = x_ump[:len(train)]
        q_x = x_ump[len(train):]
        ax.scatter(q_i[:, 0], q_i[:, 1], color=color_map[0], edgecolor='white', label=labels[0])
        ax.scatter(q_x[:, 0], q_x[:, 1], color=color_map[1], edgecolor='white', label=labels[1])
        
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.savefig(f"{dir}/dataset_{train.shape[-1]}.png")
        plt.close()

    elif schedule_plot == "stage":
        fig, axs = plt.subplots()
        data, y_r = dataset[:][0], dataset[:][1]['y']
        
        # If class_names provided, use them; otherwise try dataset method
        if class_names:
            labels = class_names
        else:
            labels = dataset.show_classes()

        x_ump = reducer.fit_transform(data)
        q_i = x_ump
        score = silhouette_score(q_i, y_r)
        added_labels = set()
        
        for ele in y_r:
            index_mask = (ele == y_r)
            if np.any(index_mask):
                current_label = labels[ele] if ele < len(labels) else str(ele)
                label = current_label if current_label not in added_labels else None
                axs.scatter(q_i[index_mask, 0],
                            q_i[index_mask, 1],
                            label=label, 
                            color=cm.get_cmap('tab10').colors[ele % 10],
                            edgecolor='white')
                if label:
                    added_labels.add(current_label)
        
        axs.axes.xaxis.set_ticklabels([])
        axs.axes.yaxis.set_ticklabels([])
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.legend(loc="upper right")
        fig.text(0.1, 0.1, f"The score is:{score:.2f}", fontsize=12, color='red', ha='left', va='bottom')
        plt.savefig(f"{dir}/StageUMAP_{data.shape[-1]}.png")
        plt.close()
       
    elif schedule_plot == "reverse":
        data_r, y_r = dataset[:][0], dataset[:][1]['y']
        data_f, y_f = synthesis_data["arr_0"], synthesis_data["arr_1"]
        
        mmd = maximum_mean_discrepancy(data_r, data_f)
        logger.info(f"The mmd score is:{mmd}")
        data_merged = np.vstack([data_r, data_f])
        fig, ax = plt.subplots()
        
        # Dynamic label creation for Real vs Fake
        # Assumes y_r and y_f indices match the class_names list
        colors = cm.get_cmap('tab10').colors
        
        q_i = reducer.fit_transform(data_merged)
        q_r = q_i[:len(y_r)]
        q_f = q_i[len(y_r):]
        
        # Plot Real Data
        for ele in np.unique(y_r):
            index_mask = (ele == y_r)
            if np.any(index_mask):
                c_name = labels_map[ele] if ele < len(labels_map) else str(ele)
                label_text = f"Real {c_name}"
                label = label_text if label_text not in added_labels else None
                
                ax.scatter(q_r[index_mask, 0],
                            q_r[index_mask, 1],
                            label=label, 
                            color=colors[ele % len(colors)],
                            edgecolor='white')
                if label:
                    added_labels.add(label_text)

        # Plot Fake Data
        for ele in np.unique(y_f):
            index_mask = (ele == y_f)
            if np.any(index_mask):
                c_name = labels_map[ele] if ele < len(labels_map) else str(ele)
                label_text = f"Fake {c_name}"
                label = label_text if label_text not in added_labels else None
                
                # Use a slightly different color or marker style for fake could be better, 
                # but following original style: offset color index or use alpha
                # Original logic: color_map[ele+2]. Here we rotate colors.
                color_idx = (ele + 2) % len(colors) 
                
                ax.scatter(q_f[index_mask, 0],
                            q_f[index_mask, 1],
                            label=label, 
                            color=colors[color_idx],
                            edgecolor='white',
                            alpha=0.5)
                if label:
                    added_labels.add(label_text)

        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.legend(loc="upper right")
        plt.savefig(f"{dir}/UMAP_plot_realvsfake_{data_r.shape[-1]}.png")
        plt.close()

    elif schedule_plot == "perturb":
        # dataset_N = Target/Reference Data
        # dataset_T = Source/Original Data
        # target = Perturbed Data (Source transformed to look like Target)
        dataset_N, dataset_T, target = dataset 
        mmd = maximum_mean_discrepancy(dataset_N, target)
        logger.info(f"The mmd score is:{mmd}")
        
        # Determine labels dynamically
        # Assuming class_names passed as [Target_Class_Name, Source_Class_Name]
        target_name = labels_map[0] if len(labels_map) > 0 else "Normal"
        source_name = labels_map[1] if len(labels_map) > 1 else "Tumor"
        
        x_merged = np.vstack([dataset_N, dataset_T, target])
        x_ump = reducer.fit_transform(x_merged)
        
        len_n = len(dataset_N)
        len_t = len(dataset_T)
        
        q_n = x_ump[:len_n] # Real Target
        q_t = x_ump[len_n : len_n + len_t] # Real Source
        q_x = x_ump[len_n + len_t:] # Perturbed Source
        
        fig, ax = plt.subplots()      
        
        # Plot 1: Real Target (Reference)
        ax.scatter(q_n[:, 0], q_n[:, 1], color='blue', edgecolor='white', label=f"Real {target_name}")
        
        # Plot 2: Real Source (Original)
        ax.scatter(q_t[:, 0], q_t[:, 1], color='orange', edgecolor='white', label=f"Real {source_name}")
        
        # Plot 3: Perturbed (Transformed)
        ax.scatter(q_x[:, 0], q_x[:, 1], color='cyan', edgecolor='white', label=f"Perturbed {source_name}")
        
        plt.legend(loc="upper right")
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



def filter_gene(real, perturb, corerate=1):
    assert real.shape == perturb.shape, f'The datashape of real and perturbed are different'
    # Calculate the difference between the corresponding gene
    logger.log(f"The real data mean {real.mean(axis=0)}")
    logger.log(f"The real data std {real.std(axis=0)}")
    logger.log(f"The perturb data mean {perturb.mean(axis=0)}")
    logger.log(f"The perturb data std {perturb.std(axis=0)}")
    differences = np.abs(real - perturb).mean(axis=0)
    logger.log(f"The differences between real and perturb data {differences} ")
    
    # Calculate the standard deviation of the differences for each gene 
    std_deviation = np.std(differences)
    logger.log(f"The standard deviation between real and perturb data data {std_deviation} ")
    # fileter columns where the difference is greater than 1 standard deviation 
    perturb_mean = differences.mean()
    logger.log(f"The mean between real and perturb data data {perturb_mean} ")
    # filter_genecoln = [( differences > perturb_mean + std_deviation) | (differences < perturb_mean - std_deviation)]
    index_array = np.arange(0,real.shape[1])
    filter_index = differences > perturb_mean + corerate * std_deviation
    perturb_perc = np.divide(differences, real.mean(axis=0))
    logger.log(f"The real data {real.mean(axis=0)[filter_index]}")
    logger.log(f"The perturb data {perturb.mean(axis=0)[filter_index]}")
    logger.log(f"The perturbation percentages between real and perturb data data {perturb_perc[filter_index]}")
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


def maximum_mean_discrepancy(X, Y, kernel_function='rbf', gamma=None):
    """
    Calculate the Maximum Mean Discrepancy (MMD) between two sets of samples.
    
    Args:
        X (numpy.ndarray): First set of samples, shape (n_samples_X, n_features).
        Y (numpy.ndarray): Second set of samples, shape (n_samples_Y, n_features).
        kernel_function (str): Kernel function to use. Options: 'rbf' (default) or 'linear'.
        gamma (float): Gamma parameter for the RBF kernel. If None, default to 1 / n_features.
    
    Returns:
        float: Maximum Mean Discrepancy between X and Y.
    """
    assert X.shape[1] == Y.shape[1], "X and Y must have the same number of features"
    
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    
    if kernel_function == 'rbf':
        kernel = lambda x, y: np.exp(-gamma * np.sum((x - y) ** 2))
    elif kernel_function == 'linear':
        kernel = lambda x, y: np.dot(x, y)
    else:
        raise ValueError("Invalid kernel function. Choose 'rbf' or 'linear'.")
    
    K_XX = np.zeros((n_samples_X, n_samples_X))
    K_YY = np.zeros((n_samples_Y, n_samples_Y))
    K_XY = np.zeros((n_samples_X, n_samples_Y))
    
    for i in range(n_samples_X):
        for j in range(n_samples_X):
            K_XX[i, j] = kernel(X[i], X[j])
        for j in range(n_samples_Y):
            K_XY[i, j] = kernel(X[i], Y[j])
    
    for i in range(n_samples_Y):
        for j in range(n_samples_Y):
            K_YY[i, j] = kernel(Y[i], Y[j])
    
    mmd = np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)
    
    return mmd



