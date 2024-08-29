import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.evaluation import MetricsDAG
from model.Actor import *
from model.Critic import *
from reward.Reward_BIC import *
from utils.record import Record
from utils.utils import ordering_matrix_pruning, from_order_to_graph
from reward.Reward_BIC_gpu import get_Reward_gpu
from utils.config import Config


class CokeModel:
    def __init__(
        self,
        acotr_config: Config,
        critic_config: Config,
        record_condfig: Config,
        reward_config: Config,
        X_full:np.ndarray,
        device: str,
    ):
        super().__init__()
        self.device = torch.device(device)

        self.actor = Actor(acotr_config).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=acotr_config.actor_lr)
        self.critic = Critic(critic_config).to(self.device)
        self.critic_optim = optim.Adam(
            self.critic.parameters(), lr=critic_config.critic_lr
        )

        self.reward = Reward(config=reward_config, inputdata=X_full)  # !CPU
        self.base_line = reward_config.base_line
        self.base_line_rate = reward_config.base_line_rate

        self.rec = Record(record_condfig)
        self.causal_matrix = None

    def trainer(
        self,
        X_full: np.ndarray,
        X_miss: list,
        initinal_graph: np.ndarray,
        trainer_config: Config,
    ) -> np.ndarray:

        adj = torch.from_numpy(initinal_graph).float().to(self.device)

        for ep in tqdm(range(trainer_config.epoch)):
            input_data = self.get_batch_data(
                trainer_config.batch_size,
                X_full,
                X_miss,
                data_dim=trainer_config.n_samples,
                random_seed=ep,
                device=self.device,
                use_x_miss=trainer_config.use_x_miss,
            )

            self.actor.train()
            self.critic.train()
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()

            enc, positions, log_softmax, errors = self.actor(input_data, adj)
            graphs = ordering_matrix_pruning(
                trainer_config.batch_size, positions, adj
            ).to(self.device)

            # * calculate rewards
            # rewards = self.reward.cal_rewards(graphs, positions).to(self.device) #! GPU

            rewards = self.reward.cal_rewards(
                graphs=graphs, positions=positions
            )  # !CPU
            rewards = torch.from_numpy(np.array(rewards)).to(self.device)
            self.base_line = self.base_line_rate * self.base_line + (
                1 - self.base_line_rate
            ) * torch.mean(rewards)

            # * update critic
            value = self.critic(enc)
            critic_loss = nn.MSELoss()((rewards - self.base_line).float(), value)
            critic_loss.backward(retain_graph=True)
            self.critic_optim.step()

            # * update actor
            td_error = rewards - self.base_line - value
            actor_loss = -(torch.mean(td_error.detach().clone() * log_softmax))
            actor_loss.backward()
            self.actor_optim.step()

            # * Record the training process
            self.rec.update_met(graph=graphs[torch.argmax(rewards).item()], ep=ep)
            self.rec.update_reward(
                reward=rewards, se_num=self.reward.must_exist_edge_errors
            )

        max_order = self.reward.update_all_scores()[0][0]
        result_graph = from_order_to_graph(max_order)  # !CPU

        self.causal_matrix = result_graph * adj.cpu().detach().numpy()
        return self.causal_matrix

    def get_batch_data(
        self,
        batch_size: int,
        X_full:np.ndarray,
        X_miss:list,
        data_dim: int,
        random_seed: int = 17,
        device: str = "cpu",
        use_x_miss=False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        X_full_batch: (bs, (num_variables,n_samples))
        X_miss_batch: (num_recipes, bs, (num_variables,n_samples))
        X_full_batch_dim: the dimension for each variables in X_full_batch, which indicates (dim, num_variables)
        X_miss_batch_dim: the dimension for each variables in X_miss_batch, which indicates (dim, num_variables)
        """
        np.random.seed(random_seed)
        device = torch.device(device)
        assert data_dim <= len(
            X_full
        ), " the full_dim is larger than the dimension of the full data"

        X_full_batch, X_miss_batch = [], None
        for _ in range(batch_size):
            select_smaples = np.random.choice(
                list([*range(0, len(X_full), 1)]), size=data_dim, replace=False
            )
            X_full_batch.append(X_full[select_smaples])
        X_full_batch = (
            torch.from_numpy(np.array(X_full_batch)).float().permute(0, 2, 1).to(device)
        )

        if use_x_miss:
            X_miss_batch = []
            for _ in range(batch_size):
                recipes_samples = []
                for rep in X_miss:
                    if len(rep) >= data_dim:
                        select_smaples = np.random.choice(
                            [*range(0, len(rep), 1)], size=data_dim, replace=False
                        )
                        recipes_samples.append(rep[select_smaples])
                X_miss_batch.append(recipes_samples)
            X_miss_batch = (
                torch.from_numpy(np.array(X_miss_batch))
                .float()
                .permute(1, 0, 3, 2)
                .to(device)
            )
            return (X_full_batch.to(device), X_miss_batch.to(device))
        else:
            return (X_full_batch.to(device), X_miss_batch)
