import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri
from rpy2.robjects.vectors import ListVector
from tqdm import tqdm

from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)   # will display errors, but not warnings

base = rpackages.importr('base')
utils = rpackages.importr('utils')
cam = rpackages.importr('CAM')
mboost = rpackages.importr('mboost')
def _pns_cam(X,d,pruneMethod=robjects.r['selGamBoost'],
             pruneMethodPars=ListVector({'(atLeastThatMuchSelected ': 0.02, 'atMostThatManyNeighbors ': 10}), output=False):
    X = robjects.r.matrix(numpy2ri.py2rpy(X), ncol=d)
    conG = robjects.r.matrix(numpy2ri.py2rpy(np.ones([d])-np.eye(d)), ncol=d)
    finalG = robjects.r.matrix(0, d, d)
    for i in tqdm(range(d), desc='Preliminary neighborhood selection'):
        parents = robjects.r.which(conG.rx(True, i + 1).ro == 1)
        lenpa = robjects.r.length(parents)[0]
        Xtmp = robjects.r.cbind(X.rx(True, parents), X.rx(True, i + 1))
        selectedPar = pruneMethod(Xtmp, k=lenpa + 1, pars=pruneMethodPars, output=output)
        finalParents = parents.rx(selectedPar)
        finalG.rx[finalParents, i + 1] = 1
    return np.array(finalG)

def pns_gam(XX,atLeastThatMuchSelected=0.02,atMostThatManyNeighbors=10):
    d = XX.shape[1]
    X2 = numpy2ri.py2rpy(XX)
    Adj = _pns_cam(X=X2,d=d,pruneMethod=robjects.r.selGamBoost,
                   pruneMethodPars=ListVector({'atLeastThatMuchSelected': atLeastThatMuchSelected, 'atMostThatManyNeighbors': atMostThatManyNeighbors}), output=False)
    return Adj
        
class Adjacency_Matrix():
    
    def __init__(self,num_var:int,is_syn:bool,var_index:list):
        self.num_var = num_var
        self.adj = np.ones([num_var, num_var]) - np.eye(num_var)
        self.must_exist_edges_adj  = None
        self.max_ordering = []
        self.is_syn = is_syn
        self.var_index = {name:idx for idx,name in enumerate(var_index)} # {var_name:var_index}

    def get_initinal_graph(self,
                           is_chronological_order : bool = False,
                           is_pns_gam : bool = False,
                           is_domain_knowledge : bool = False,
                           chronological_order_info : dict = {},
                           X_full : np.ndarray = None,
                           domain_edges : list = [str,str],
                           ) -> np.ndarray:
        
        if is_chronological_order:
            print("[USE] chronological_order")
            self.adj_chro = self.prun_by_constraint(self.adj,chronological_order_info)
            self.adj = self.adj_chro

        if is_pns_gam:
            print("[USE] pns gam")
            adj_gam = np.array(pns_gam(X_full))
            self.adj = self.adj*adj_gam
            
        if is_domain_knowledge:
            print("[USE] domain knowledge")
            self.adj = self.modify_by_domain_knowledge(self.adj,domain_edges)
            self.must_exist_edges_adj = self.get_must_exist_edge_adj(domain_edges[0])
        return self.adj

    def prun_by_constraint(self,adj:np.ndarray=None,time_info:dict=None) -> np.ndarray:
        idx_for_group = { i:g for g,idx_list in time_info.items() for i in idx_list}
        for i in range(self.num_var):
            for j in range(self.num_var):
                if idx_for_group[i] > idx_for_group[j]:
                    adj[i,j] = 0
        if self.is_syn:
            for i in range(self.num_var):
                for j in range(self.num_var):
                    if i > j:
                        adj[i,j] = 0
        return adj
    
    def modify_by_domain_knowledge(self,adj:np.ndarray,domain_edges):
        exist_flag = domain_edges[0] != ""
        delete_flag = domain_edges[1] != ""
        if exist_flag:
            must_exist_edge = [edges.split(',') for edges in domain_edges[0].split(';')]
            for [src_name,trg_name] in must_exist_edge:
                src_idx,trg_idx = self.var_index[src_name], self.var_index[trg_name]
                adj[src_idx,trg_idx] = 1
        if delete_flag:
            must_delete_edges = [ edges.split(',') for edges in domain_edges[1].split(';')]
            for [src_name,trg_name] in must_delete_edges:
                src_idx,trg_idx = self.var_index[src_name], self.var_index[trg_name]
                adj[src_idx,trg_idx] = 0
        return adj
        
    def get_must_exist_edge_adj(self,must_exist_edge) -> None:
        if must_exist_edge == "":
            return None
        else:
            must_exist_edge_list = [edges.split(',') for edges in must_exist_edge.split(';')]
            adj_exist = np.zeros([self.num_var,self.num_var])
            for [src_name,trg_name] in must_exist_edge_list:
                src_idx,trg_idx = self.var_index[src_name], self.var_index[trg_name]
                adj_exist[src_idx,trg_idx] = 1
        return adj_exist

