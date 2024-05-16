from torch.utils.data import DataLoader, Dataset 
from sklearn.datasets import make_s_curve 
import torch as th 
from . import logger 
import pandas as pd 
import os 
import numpy as np 
import blobfile as bf
import random
SEED = 1234



def load_data(
    *,data_dir=None, 
    class_cond=False,
    data_filter="replace",
    gene_selection=None,
    dge = False,
    gene_set = None,
):
    """ 
    For a dataset, create a generate over (seq, kwars) pairs

    Each sequence is an NCX float tensor, and the kwargs dict contains zero or 
    more keys, each of which map to a batched Tensor of their own. 

    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensor of class labels.
    
    :param data_dir: a dataset directory. -- if data dir is None, load data from sklearn library
    :param batch_size: the batch size of each returned pair.
    :prarm class_cond: if True, include a "y" key in returned dicts for class
                       labels. If classes are not available and this is True, an
                    execption will be raised. 
    """

    # if data_dir == None:
    #     s_curve, _ = make_s_curve(10**3,noise=0.1)
    #     s_curve = s_curve[:,[0,2]]/10.0
    #     dataset = th.Tensor(s_curve).float()
    #     logger.log(f"Loading dataset from example data and the shape of the dataset is: {dataset.size()}")
    #     dataset = SklearnDataset(dataset)
    #     return dataset 
    
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    all_files = _list_files_recursively(data_dir)
    # set the condition later
    # logger.debug(f"The information of {gene_selection} -- datasets")
    dataset = CustomGeneDataset(all_files[0],
                                all_files[1],
                                gene_set = gene_set,
                                transform= GeneDataTransform(),
                                target_transform=GeneLabelTransform(),
                                scaler=True,
                                filter=data_filter,
                                random_selection=(GeneRandom(seed=SEED,features=gene_selection) if gene_selection else None),
                                dge = (Genedifferential() if dge else None), 
                                class_cond =class_cond,
    )
    
    
    
    # if gene_set 
    # if gene_selection is not None:
    #     dataset = CustomGeneDataset(all_files[0],
    #                                 all_files[1],
    #                                 transform= GeneDataTransform(),
    #                                 target_transform=GeneLabelTransform(),
    #                                 scaler=True,
    #                                 filter=data_filter,
    #                                 random_selection=GeneRandom(seed=SEED,features=gene_selection),
    #     )
    # elif dge:
    #     dataset = CustomGeneDataset(all_files[0],
    #                                 all_files[1],
    #                                 transform= GeneDataTransform(),
    #                                 target_transform=GeneLabelTransform(),
    #                                 scaler=True,
    #                                 filter=data_filter,
    #                                 dge=Genedifferential(),
    #                                 class_cond =class_cond,
    #     )
    
    # else:
    #     dataset = CustomGeneDataset(all_files[0],
    #                                 all_files[1],
    #                                 transform= GeneDataTransform(),
    #                                 target_transform=GeneLabelTransform(),
    #                                 scaler=True,
    #                                 filter=data_filter,
    #     )
    # logger.log(f"After data pre-processing, the dataset contains {dataset[0]}")
    logger.log(f"After data pre-processing, the dataset contains {dataset[:][0].shape[-1]} gene.")
    #logger.log("The gene selection is fixed random. Have not set a bool value for examine whether fixed random or true random")
    return dataset


def data_loader(dataset, batch_size=32,deterministic=False):
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, 
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, 
        )
    
    return loader
    



def _list_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir), key=lambda x: (x[0].isupper(), x)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["txt"]:
            results.append(full_path)
        elif "." in entry and ext.lower() in ["gmt"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_files_recursively(full_path))
    return results



class CustomGeneDataset(Dataset): 
    def __init__(
        self, 
        genepath="data.txt",
        labelpath="label.txt", 
        gene_set = None,
        transform=None, 
        filter = None, 
        scaler = None, 
        target_transform=None, 
        random_selection = None,
        dge = None,
        class_cond = False,
    ):
        assert os.path.exists(genepath), "gene path: {} does not exist.".format(genepath)
        logger.log(f"reading input data from {os.path.basename(genepath)}") 
        # read the gene expression 
        df = pd.read_csv(genepath, sep='\t', index_col=0)
        logger.log(f"loaded input data has {df.shape[1]} genes, {df.shape[0]} samples")
        if gene_set is not None:
            if gene_set.lower().endswith('.tsv'):
                geneset = pd.read_csv(gene_set, delimiter='\t')
                geneset = list(set(geneset["#node1"]))
                df = df[df.columns.intersection(geneset)]
            else: # else include gmt and pure -- need to specific the case in the future to avoid protential error 
                geneset = read_file(gene_set)
                logger.log(f"The input gene set is {geneset['gene_set']}, contains {len(geneset['genes'])} genes")
                # logger.debug(f"The intersection is : {df.columns} -- dataset")
                # logger.debug(f"The intersection is : {df.columns} -- dataset")
                df = df[df.columns.intersection(geneset['genes'])]
            logger.log(f"loaded selected data has {df.shape[1]} genes, {df.shape[0]} samples")
            
        gene_features = df.values
    
        assert os.path.exists(labelpath), "gene label path: {} does not exist.".format(labelpath)
        # read the gene labels
        df_labels= pd.read_csv(labelpath, sep='\t', header=None)
        classes = sorted(set(df_labels[1]))
        logger.info(f"loaded input data has {len(classes)} classes")
        self.classes = classes
       
        if transform:
            gene = transform(gene_features, scaler, filter)
            
        if target_transform:
            label = target_transform(df_labels[1].values, classes)
        
        if random_selection:
            gene = random_selection(gene)
        if dge:
            gene = dge(gene)

        self.gene = gene
        self.label = label

    def __len__(self):
        return len(self.gene)
    
    def __getitem__(self,idx):
        
        out_dict = {}
        if self.classes:
            out_dict["y"] = np.array(self.label[idx], dtype=np.int64)
        return np.array(self.gene[idx], dtype=np.float32), out_dict


class GeneDataTransform():
    """
    For a dataset, data pre-process.

    :param sample: data array.
    :param scaler: If True, normalize data sample.
    :param filter: a function which drops nan values or replace.
    
    """    
    def __call__(self, sample, scaler= None, filter = None,):
        
        # logger.debug(f"gene has {sample.shape[1]} genes, {sample.shape[0]} samples before pre-processing  -- mrna")
        if filter != None:
            if filter == "replace": 
                # filter the nan value to -inf
                # log2-inf turns the gene expression to 0. 
                mask_sample = sample[~np.isnan(sample)]
                mask_nan = -np.inf
                sample = np.nan_to_num(sample, nan = mask_nan)
            elif filter == "drop":
                # drop the nan value by columns
                sample = sample[:, ~np.isnan(sample).any(axis=0)]
                """ reduce genes from 19738 to 12811 """
            # simple trasformation from log2(n) to log2(n+1)
            sample = np.log2(np.exp2(sample)+1)
        # unet upsample cannot work with odd feature size. 
        # if features are not integer multiples of 128, drop columns with minim feature value. 
        #     while sample.shape[1] % 128 != 0:
        #         column_sums = np.sum(sample, axis=0)
        #         min_sum_col_index = np.argmin(column_sums)
        #         # drop the identified column 
        #         sample = np.delete(sample, min_sum_col_index, axis=1)
        if scaler:
            """ MaxAbsScaler -- can be heavily influenced by outliers. """
            # mins, maxs = np.amin(sample, axis=0)[0], np.amax(sample, axis=0)[0]        
            # scaler_ = np.maximum(np.abs(mins),np.abs(maxs))
            # sample = np.divide(sample, scaler_)
                   
            """ Min-Max scaler"""
            mins, maxs = sample.min(), sample.max()
            sample = 2 * (sample - mins) / (maxs - mins) - 1
        # comment out for test experiment     
        # sample = sample[:,np.newaxis,:]
        return sample


class GeneLabelTransform():
    """
    For a dataset, data pre-process.

    :param label: label array.
    
    """    
    def __call__(self, label, classes):
        # convert the label to integer 
        label = np.array([classes.index(i) for i in label])
        return label


class GeneRandom():
    def __init__(self, seed=None,features = 100):
        self.seed = seed 
        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.features = features 
        if isinstance(self.features, str):
            self.features = int(self.features)
  
    def __call__(self, sample):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        # random select the gene 
        if self.features <= sample.shape[-1]:
            idx = np.random.randint(0,sample.shape[-1], self.features)
            return sample[:,idx]
        return sample [:,:]
        # return sample[:,:,idx]
        



class SklearnDataset(Dataset): 
    def __init__(self,dataset,label=None,classes=None):
        logger.info("reading input data from sklearn datasets") 
    
        self.data = dataset[:,np.newaxis,:]
        self.label = label
        self.classes = classes
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        
        out_dict = {}
        if self.classes is not None:
            out_dict["y"] = np.array(label[idx], dtype=np.int64)
        return np.array(self.data[idx], dtype=np.float32), out_dict


class Genedifferential():
    def __init__(self, index = [9583,3605]):
        self.index = index
  
    def __call__(self, sample):
        return sample[:,self.index]
        # return sample[:,:,self.index]

    

def datascalar(dataset):

    mins, maxs = dataset.min(), dataset.max()
    return mins, maxs 


def read_file(filename):
    gene_sets = {}
    if os.path.splitext(filename)[1] == '.gmt':
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                gene_set_name = parts[0]
                # description = parts[1]
                genes = parts[2:]
                gene_sets = {'gene_set':gene_set_name, 'genes': genes}
    else:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                gene_set_name = parts[0]
                genes = parts[1:]
                gene_sets = {'gene_set':gene_set_name, 'genes': genes}
    return gene_sets