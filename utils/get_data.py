import pandas as pd
import numpy as np
import torch

class DataLoader():

    '''
    parameters explanation
    data: pd.DataFrame
    batch_size: 
    no_use_col
    lower_number_sammples
    
    '''
    def __init__(self, 
                 datasetpath:str,
                 labelpath:str,
                 no_use_col:str,
                 lower_number_samples:int=10,
                 sorted = False,
                 is_syn = False,
                 chronological_order = False,
                 use_x_miss = False
                 ):

        if sorted:
            self.data = pd.read_csv(datasetpath).sort_index(axis=1)
        else:
            self.data = pd.read_csv(datasetpath)
        no_use_col = no_use_col.split(",") if no_use_col != "" else []
        self.label = pd.read_csv(labelpath)
        self.data = self.data.drop(columns=no_use_col)
        self.node_list = self.data.columns.to_list()
        self.num_variables = len(self.node_list)
        self.is_syn = is_syn
        self.lower_number_samples = lower_number_samples
        self.use_x_miss = use_x_miss
        self.recipe = self.get_recipe(self.data)
        self.X_full, self.X_miss_dataset = self.get_data(self.data)
        self.true_dag = self.get_label_adj(
            self.num_variables, self.label, self.node_list
        )
        self.chronological_order_info = None
        if chronological_order:
            self.chronological_order_info = self.get_chronological_order(self.node_list)

        # print("Finish getting data")

    def get_chronological_order(self,col_list:list):
        time_sequence_info = {}
        for i,col in enumerate(col_list):
            cols = col.split('_')
            if not self.is_syn:
                group_id = int(cols[1][2:]) -1 #real data(2)
            else:
                group_id = int(cols[0][1:]) #real_data(1)
            if group_id in time_sequence_info:
                time_sequence_info[group_id].append(i)
            else:
                time_sequence_info[group_id] = [i]
        return time_sequence_info

    def get_recipe(self,df:pd.DataFrame) -> dict:
        recipes = {}
        for i,row in df.iterrows():
            non_na_cols = tuple(list(row[~row.isnull()].index.to_list()))
            if non_na_cols in recipes:
                recipes[non_na_cols].append(i)
            else:
                recipes[non_na_cols] = [i]
        return recipes

    def get_X_miss_idx_by_reipces(self,num_samples,recipes,all_columns):
        if all_columns is not None: del recipes[all_columns]
        for cols_tuple, sample_list in recipes.copy().items():
            if len(sample_list)<num_samples:
                del recipes[cols_tuple]
        return recipes 

    def get_data(self,data):
        X_full, X_miss_dataset = None, None
        _recipes = self.recipe.copy()
        assert tuple(data.columns) in _recipes , "There are no X_full  is not in the recipe"
        assert len(_recipes[tuple(data.columns)]) >= self.lower_number_samples, "The number of samples in X_full is less than the lower_number_samples"

        X_full_idx = {tuple(data.columns):_recipes[tuple(data.columns)]}
        X_full = data.iloc[X_full_idx[tuple(data.columns)]].to_numpy()  
        all_cols = tuple(data.columns)
        if self.use_x_miss:
            X_miss_idx = self.get_X_miss_idx_by_reipces(self.lower_number_samples,_recipes,all_cols)
            X_miss_dataset = [data.iloc[v].to_numpy()for k,v in X_miss_idx.items()]
        return X_full, X_miss_dataset

    def get_label_adj(self, num_variables:int=10, label:pd.DataFrame=None, node_list:list=None):
        adj_true = np.zeros((num_variables,num_variables))
        for [start,end] in label.values.tolist():
            adj_true[node_list.index(start),node_list.index(end)] = 1
        return adj_true
