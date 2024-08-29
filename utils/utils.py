import torch
import numpy as np
from utils.evaluation import MetricsDAG

def ordering_matrix_pruning(bs, positions, _adj) -> torch.Tensor:
    if positions.device != torch.device('cpu'): positions = positions.cpu().detach().numpy()
    if _adj.device != torch.device('cpu'): _adj = _adj.cpu().detach().numpy()
    else: _adj = _adj.numpy()
    ordering_matrixs = []
    for j in range(bs):
        _matrix = from_order_to_graph(positions[j])
        ordering_matrixs.append(_matrix * _adj)
    return torch.from_numpy(np.array(ordering_matrixs)).float()


def from_order_to_graph(true_position) -> np.ndarray:
    d = len(true_position)
    zero_matrix = np.zeros([d, d])
    for n in range(d - 1):
        row_index = true_position[n]
        col_index = true_position[n + 1:]
        zero_matrix[row_index, col_index] = 1
    return zero_matrix

def output_result(causal_graph,true_graph,_time=0,var=0,):
    met = MetricsDAG(causal_graph, true_graph)
    result = 'type:{},time:{},fdr:{},tpr:{},shd:{},F1:{},precision:{},recall:{}\n'.format('CBORL-' + str(var), _time, met.metrics['fdr'], met.metrics['tpr'], met.metrics['shd'], met.metrics['F1'], met.metrics[
        'precision'], met.metrics['recall'])
    print(*(result.split(',')),sep='\n') 
    print("---------------------------")
    return result