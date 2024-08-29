import torch
from utils.evaluation import MetricsDAG
from aim import Run,Image
import  numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Record():
    def __init__(self,config):
        
        '''
        max_rewards_per_batch : (bs,) list of rewards, each element is the max reward in a batch
        mean_rewards_per_batch : (bs,) list of rewards, each element is the mean reward in a batch
        '''
        self.max_rewards_per_batch = []
        self.mean_rewards_per_batch = []
        
        self.mets = {
            'fdr':[],
            'tpr':[],
            'shd':[],
            'F1':[],
            'precision':[],
            'recall':[]
        }
        self.max_rewards = []
        self.max_reward = float('-inf') #TODO 有 maximun reward 的 ordering graph 可能不是最好的 graph
        self.record_aim = config.record_aim
        self.true_dag = config.true_dag
        if self.record_aim:
            
            self.run = Run(experiment="COKE")
            self.run["hparams"] = {k: v for k, v in config.all_config.__dict__.items() if k != "device" and k!="true_dag"}

    def update_batch(self,max:float,mean:float):
        self.max_rewards_per_batch.append(max)
        self.mean_rewards_per_batch.append(mean)
    def update_global(self,_max_reward:float):
        self.max_reward = _max_reward if _max_reward > self.max_reward else self.max_reward
        self.max_rewards.append(self.max_reward)
    def update_reward(self,reward:torch.Tensor,ep=0,se_num=0):
        _max_reward = torch.max(reward).item()
        _mean_reward = torch.mean(reward).item()
        if self.record_aim:
            self.run.track({
                "max_reward": _max_reward,
                "mean_reward": _mean_reward,
                "Strong_Edge":se_num,
            }, epoch=ep)
        self.update_batch(_max_reward,_mean_reward)
        self.update_global(_max_reward)
        
        
    def update_met(self,graph,ep):
        if type(graph)==torch.Tensor:
            if "cuda" in graph.device.type:
                graph = graph.cpu().detach().numpy()
            else:
                graph = graph.detach().numpy()
        assert type(graph)==np.ndarray, "The type of graphs should be np.ndarray"
    
        met = MetricsDAG(graph, self.true_dag)
        for k in ['fdr','tpr','shd','F1','precision','recall']:
            self.mets[k].append(met.metrics[k])
            if self.record_aim:
                self.run.track({
                    k: met.metrics[k],
                }, epoch=ep)
    def update_img_graph(self,graph:np.array,ep):
        unit8_g = graph.astype(np.uint8)
        unit8_g  = np.where(unit8_g == 1,0,255)

        if self.record_aim:
            self.run.track(Image(unit8_g), name='ordering graph', step=ep)