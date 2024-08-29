import time 
from model.COKE import CokeModel
from utils.adjacency import Adjacency_Matrix
from utils.get_data import DataLoader
from utils.config import get_parser,Config
from utils.utils import output_result
import pandas as pd

import os 

def main():

    data = DataLoader(
        datasetpath=config.datapath,
        labelpath=config.labelpath,
        no_use_col=config.no_use_cols,
        lower_number_samples=config.n_samples,
        is_syn=config.is_synthetic,
        chronological_order=config.chronological_order,
        use_x_miss = config.use_x_miss,

    )

    adjacency = Adjacency_Matrix(
        num_var=data.num_variables, is_syn=config.is_synthetic, var_index=data.node_list
    )

    X_full = data.X_full
    X_miss = data.X_miss_dataset
    initinal_graph = adjacency.get_initinal_graph(
        is_chronological_order=config.chronological_order,
        is_pns_gam=config.pns_gam,
        is_domain_knowledge=config.domain_knowledge,
        chronological_order_info=data.chronological_order_info,
        X_full=X_full if config.pns_gam else None,
        domain_edges=[config.must_exist_edges, config.must_delete_edges],
    )

    config.num_variables = data.num_variables
    config.true_dag = data.true_dag

    print("----- before training ----- ")
    result = output_result(initinal_graph, config.true_dag, var=config.num_variables)
    coke = CokeModel(
        acotr_config=Config(config, "actor"),
        critic_config=Config(config, "critic"),
        reward_config=Config(config, "reward", adjacency.must_exist_edges_adj),
        record_condfig=Config(config, "record"),
        device=config.device,
        X_full=X_full,
    )

    begin_time = time.time()
    time_tuple= time.localtime(begin_time)
    folder =time.strftime("%m-%d-%H-%M-%S",time_tuple)
    os.makedirs( "result/"+folder, exist_ok=True)

    causal_graph = coke.trainer(
        X_full=X_full,
        X_miss=X_miss,
        initinal_graph=initinal_graph,
        trainer_config=Config(config, "trainer"),
    )
    end_time = time.time()
    
    print("----- after training ----- ")
    result = output_result(
        causal_graph,
        config.true_dag,
        _time=end_time - begin_time,
        var=config.num_variables,
    )
    file_handle = open("result/result.csv", "a")
    file_handle.write(f"folder:{folder},data:{config.datapath}"+result)
    file_handle.close()

    causal_graph_df = pd.DataFrame(causal_graph,columns=data.node_list,index=data.node_list)    
    causal_graph_df.to_csv(f"result/{folder}/causal_graph.csv")
    # print(causal_graph)
    causal_graph_list = pd.DataFrame(
        [[data.node_list[src],data.node_list[dst]] for src in range(config.num_variables) for dst in range(config.num_variables) if causal_graph[src][dst] == 1],
        columns = ['from','to'],
    )
    causal_graph_list.to_csv(f"result/{folder}/causal_graph_list.csv",index=False)

if __name__ == "__main__":
    config = get_parser()
    if config.debugger:
        config.use_x_full = True
        config.use_x_miss = True
        config.synthetic = 'real'
        config.datapath = 'Synthetic_HECSL_dataset/_10000_50_20_21_0_17_4/X_miss(0.48)_rep(75)_xfull(100).csv'
        config.labelpath = 'Synthetic_HECSL_dataset/_10000_50_20_21_0_17_4/arcs_miss(0.48)_rep(75)_xfull(100).csv'
        config.no_use_cols =  []
        config.epoch=2
        config.pns_gam = True
        config.alpha_full = 1
        config.alpha_miss=1
        config.chronological_order = True
        config.must_exist_edges = "p8_ms1_m18,p9_ms1_m21;p7_ms0_m16,p16_ms2_m39;p6_ms2_m15,p18_ms0_m42;p5_ms0_m12,p9_ms2_m22;p3_ms3_m9,p12_ms0_m30;p3_ms0_m6,p19_ms0_m43;p1_ms1_m2,p9_ms3_m23;p0_ms0_m0,p9_ms6_m26;p2_ms1_m5,p3_ms3_m9;p9_ms5_m25,p16_ms0_m37"
        config.must_delete_edges = "p0_ms0_m0,p1_ms2_m3;p1_ms1_m2,p3_ms1_m7"
    main()
