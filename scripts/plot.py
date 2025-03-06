import os
import argparse
import numpy as np
import torch as th
from omegaconf import OmegaConf
from diffusion import dist_util, logger
from diffusion.script_util import showdata, find_model, create_model_and_diffusion
from diffusion.datasets import load_data, data_loader
import datetime 
from torch.utils.data import DataLoader,TensorDataset
from diffusion.mlp import create
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


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
    logger.log(f"Config information:{config}")
    # load data 
    train_data, test_data = load_data(train_path = config.data.train_path,
                    train_label_path = config.data.train_label_path,
                    test_path = config.data.test_path,
                    test_label_path = config.data.test_label_path,
                    gene_selection = config.model.feature_size,
                    class_cond=config.model.class_cond,
                    gene_set = args.gene_set,
    )
    loader = data_loader(train_data,
                    batch_size=config.train.batch_size,
                    deterministic=True,            
    ) 
    # train_data, test_data = load_data(data_dir = config.data.data_dir,
    #                 gene_selection = config.model.feature_size,
    #                 class_cond=config.model.class_cond,
    #                 gene_set = args.gene_set,
    # )
    # balance the train and test data 
    # train_N, train_T = balance_sample_screen(train_data)
    # test_N, test_T = balance_sample_screen(test_data)
    # logger.debug(f"The shape of trian data is {train_data[:][0].shape}")
    # logger.log("Plot the original dataset with UMAP")
    # showdata(train_data,dir = args.dir_out, schedule_plot = "origin",n_neighbors =config.umap.n_neighbors,min_dist=config.umap.min_dist)     
    # logger.log("Plot the forward diffsuion process with UMAP")
    # showdata(dataset,
    #          dir = dir_out,
    #          schedule_plot="forward",
    #          diffusion=diffusion, 
    #          num_steps = config.diffusion.diffusion_steps,
    # )
    if args.plot == "sample":
        # load the fake data 
        file_path = args.sample_path
        data_fake = np.load(file_path)

        logger.log("Plot the synthesis data with UMAP")
        out_path = os.path.join(args.dir_out,datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        showdata(test_data,dir = out_path, schedule_plot = "reverse", synthesis_data=data_fake,n_neighbors =config.umap.n_neighbors,min_dist=config.umap.min_dist)
    elif args.plot == "origin":
        logger.log("Plot the original data with UMAP")
        out_path = os.path.join(args.dir_out,datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        showdata([train_data, test_data],dir = out_path, schedule_plot = "origin",n_neighbors =config.umap.n_neighbors,min_dist=config.umap.min_dist)
    """
    # Create an SVM classifier
    svm = SVC(kernel='linear')
    for X_batch, y_batch in loader:
        X_batch = X_batch.numpy()
        y_batch = y_batch['y'].numpy()
        svm.fit(X_batch, y_batch)
    # Make predictions on the test set
    test_loader = data_loader(test_data,
                    batch_size=config.train.batch_size,            
    ) 
    y_pred = []
    y_true = []
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.numpy()
        y_batch = y_batch['y'].numpy()
        y_pred.extend(svm.predict(X_batch))
        y_true.extend(y_batch)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_true, y_pred)
    logger.log(f'The testa dataset Accuracy: {100 *accuracy:.2f}%')
    generated_data,generated_labels = data_fake['arr_0'],data_fake['arr_1']
    # generated_data = th.tensor(generated_data, dtype=th.float32)
    # generated_labels = th.tensor(generated_labels, dtype=th.long)
    y_pred = svm.predict(generated_data)
    accuracy = accuracy_score(generated_labels, y_pred)
    logger.log(f'The synthesis dataset Accuracy: {100 * accuracy :.2f}%')
    """
    logger.log("plot complete...")



def create_config():
    
    defaults = {
        "data":{
            "data_dir": None,
            "dir_out": "results",
            "gene_selection": None,
            # "samples":124,
            "drop_fraction":0,
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
        },
    }
    # defaults.update(model_and_diffusion_defaults())
   
    return defaults  



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/random/mrna_256.yaml")
    parser.add_argument("--dir_out", type=str, default="results/")
    parser.add_argument("--dir", type=str, default="log/")
    parser.add_argument("--sample_path", type=str, default=None)
    parser.add_argument("--gene_set", type=str, default="Random")
    parser.add_argument("--plot", type=str, default="sample")
    args = parser.parse_args()
    main(args)