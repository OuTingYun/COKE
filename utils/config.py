import argparse
import torch

def get_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--actor_lr', type=float, default=0.0011, help='actor learning rate')
    parser.add_argument('--add_error', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha value')
    parser.add_argument('--base_line', type=int, default=-1, help='base line')
    parser.add_argument('--base_line_rate', type=float, default=0.99, help='base line rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--chronological_order', action='store_true', default=False)
    parser.add_argument('--critic_lr', type=float, default=0.0041, help='critic learning rate')
    parser.add_argument('--datapath', type=str, default=None, help='dataset path')
    parser.add_argument('--debugger', action='store_true', default=False, help='debugger')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--epoch', type=int, default=2200, help='number of epochs')
    parser.add_argument('--labelpath', type=str, default=None, help='label path')
    parser.add_argument('--med_w', type=float, default=1.0, help='med_w')
    parser.add_argument('--med_w_flag', action='store_true', default=False, help='med_w_flag')
    parser.add_argument('--must_delete_edges', type=str, default="", help='src1,dst1;src1,dst1;...')
    parser.add_argument('--must_exist_edges', type=str, default="", help='src1,dst1;src1,dst1;...')
    parser.add_argument('--n', type=int, default=10, help='least of samples in recipes')
    parser.add_argument('--n_samples', type=int, default=40, help='input features')
    parser.add_argument('--nblocks', type=int, default=2, help='number of blocks')
    parser.add_argument('--nheads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--no_use_cols', type=str, default="", help='list of columns to exclude')
    parser.add_argument('--pns_gam', action='store_true', default=False)
    parser.add_argument('--panelty_ratio', type=float, default=0.1, help='panelty ratio of rmust existed edges')
    parser.add_argument('--record_aim', action='store_true', default=False, help='record aim')
    parser.add_argument('--reg_type', type=str, default='LR', help='regularization type')
    parser.add_argument('--score_type', type=str, default='BIC', help='score type, BIC: Difference is same') 
    parser.add_argument('--sem_type', type=str, default=None, help='sem_type')
    parser.add_argument('--synthetic', type=str, default='synthetic', help='synthetic data type')
    parser.add_argument('--theta_full', type=float, default=0)
    parser.add_argument('--theta_miss', type=float, default=0)
    parser.add_argument('--use_x_full', action='store_true', default=False)
    parser.add_argument('--use_x_miss', action='store_true', default=False)

    config = parser.parse_args()

    config.domain_knowledge = (
        True
        if config.must_exist_edges != "" or config.must_delete_edges != ""
        else False
    )
    config.is_synthetic =  True if config.synthetic == "synthetic" else False
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEVICE]",config.device)
    print(f"[USE]",config.device)
    assert config.use_x_full, "COKE doesn't suppose for use_x_full=False"
    assert config.n < config.n_samples, "n should be less than n_samples"

    return config


class Config:
    def __init__(self, config, name, must_exist_edges_adj=None) -> None:
        if name == "actor":
            self.actor_lr = config.actor_lr
            self.alpha = config.alpha
            self.theta_full = config.theta_full
            self.theta_miss = config.theta_miss
            self.batch_size = config.batch_size
            self.dropout = config.dropout
            # self.hid_features = config.hid_features
            self.n_samples = config.n_samples
            self.nblocks = config.nblocks
            self.nheads = config.nheads
            self.num_variables = config.num_variables
            self.use_x_full = config.use_x_full
            self.use_x_miss = config.use_x_miss
        elif name == "critic":
            self.critic_lr = config.critic_lr
            self.n_samples = config.n_samples
            self.num_variables = config.num_variables
        elif name == "reward":
            self.alpha = config.alpha
            self.base_line = config.base_line
            self.base_line_rate = config.base_line_rate
            self.n_samples = config.n_samples
            self.knowledge = config.domain_knowledge
            self.med_w = config.med_w
            self.med_w_flag = config.med_w_flag
            self.must_exist_edges_adj = must_exist_edges_adj
            self.nblocks = config.nblocks
            self.nheads = config.nheads
            self.num_variables = config.num_variables
            self.pns_gam = config.pns_gam
            self.reg_type = config.reg_type
            self.score_type = config.score_type
            self.panelty_ratio = config.panelty_ratio
        elif name == "trainer":
            self.batch_size = config.batch_size
            self.epoch = config.epoch
            self.n_samples = config.n_samples
            self.num_variables = config.num_variables
            self.reg_type = config.reg_type
            self.score_type = config.score_type
            self.synthetic = config.synthetic
            self.use_x_miss = config.use_x_miss
        elif name == "record":
            self.record_aim = config.record_aim
            self.true_dag = config.true_dag
            self.all_config = config
