"""
PPO算法解决倒立摆问题 - 面向对象版本

本文件实现了使用PPO（Proximal Policy Optimization）算法在CartPole-v1环境中训练智能体。
代码采用面向对象编程，提高可读性和可维护性。
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import rl_utils


@dataclass
class PPOConfig:
    """PPO算法配置类"""
    # 环境配置
    env_name: str = 'CartPole-v1'
    seed: int = 0
    
    # 网络配置
    hidden_dim: int = 128
    
    # 训练配置
    num_episodes: int = 100
    actor_lr: float = 1e-3
    critic_lr: float = 1e-2
    
    # PPO算法参数
    gamma: float = 0.98  # 折扣因子
    lmbda: float = 0.95  # GAE参数
    epochs: int = 10     # 每次更新的轮数
    eps: float = 0.2      # PPO截断参数
    
    # 设备配置
    device: Optional[torch.device] = None
    
    # 可视化配置
    enable_visualization: bool = False
    save_plots: bool = True
    plot_filename: str = 'ppo_training_results.png'
    
    def __post_init__(self):
        """初始化设备"""
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    """策略网络（Actor）：输出动作概率分布"""
    
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        """
        初始化策略网络
        
        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
            action_dim: 动作维度
        """
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入状态 [batch_size, state_dim]
            
        Returns:
            动作概率分布 [batch_size, action_dim]
        """
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(nn.Module):
    """价值网络（Critic）：估计状态价值函数"""
    
    def __init__(self, state_dim: int, hidden_dim: int):
        """
        初始化价值网络
        
        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
        """
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入状态 [batch_size, state_dim]
            
        Returns:
            状态价值 [batch_size, 1]
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPOAgent:
    """PPO智能体类，采用截断方式（Clipped PPO）"""
    
    def __init__(self, config: PPOConfig, state_dim: int, action_dim: int):
        """
        初始化PPO智能体
        
        Args:
            config: PPO配置对象
            state_dim: 状态维度
            action_dim: 动作维度
        """
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 创建网络
        self.actor = PolicyNet(state_dim, config.hidden_dim, action_dim).to(config.device)
        self.critic = ValueNet(state_dim, config.hidden_dim).to(config.device)
        
        # 创建优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
    
    def take_action(self, state: np.ndarray) -> int:
        """
        根据当前策略选择动作（带探索）
        
        Args:
            state: 当前状态
            
        Returns:
            选择的动作索引
        """
        # 将numpy数组转换为tensor，避免警告
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32)
        state_tensor = torch.from_numpy(np.array([state], dtype=np.float32)).to(self.config.device)
        probs = self.actor(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict: Dict[str, List]) -> None:
        """
        PPO算法核心更新函数
        
        Args:
            transition_dict: 包含一条完整轨迹的字典
                - states: 状态序列
                - actions: 动作序列
                - rewards: 奖励序列
                - next_states: 下一状态序列
                - dones: 终止标志序列
        """
        # 转换为tensor（使用numpy数组先转换，避免警告）
        states = torch.from_numpy(np.array(transition_dict['states'], dtype=np.float32)).to(self.config.device)
        actions = torch.from_numpy(np.array(transition_dict['actions'])).view(-1, 1).to(self.config.device)
        rewards = torch.from_numpy(np.array(transition_dict['rewards'], dtype=np.float32)).view(-1, 1).to(self.config.device)
        next_states = torch.from_numpy(np.array(transition_dict['next_states'], dtype=np.float32)).to(self.config.device)
        dones = torch.from_numpy(np.array(transition_dict['dones'], dtype=np.float32)).view(-1, 1).to(self.config.device)
        
        # 计算TD目标和TD误差
        td_target = rewards + self.config.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        
        # 计算优势函数（GAE）
        advantage = rl_utils.compute_advantage(
            self.config.gamma, 
            self.config.lmbda, 
            td_delta.cpu()
        ).to(self.config.device)
        
        # 保存旧策略的对数概率
        old_log_probs = torch.log(
            self.actor(states).gather(1, actions)
        ).detach()
        
        # 多轮更新
        for _ in range(self.config.epochs):
            # 计算当前策略的对数概率
            log_probs = torch.log(self.actor(states).gather(1, actions))
            
            # 计算重要性采样比率
            ratio = torch.exp(log_probs - old_log_probs)
            
            # PPO的Clipped Surrogate Objective
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.config.eps, 1 + self.config.eps) * advantage
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            
            # Critic损失
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            # 反向传播和更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


class CartPoleEnvironment:
    """CartPole环境封装类"""
    
    def __init__(self, config: PPOConfig):
        """
        初始化环境
        
        Args:
            config: PPO配置对象
        """
        self.config = config
        self.env = gym.make(config.env_name, render_mode="rgb_array" if config.enable_visualization else None)
        self._set_seed()
        
        # 获取环境信息
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
    
    def _set_seed(self) -> None:
        """设置随机种子"""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.env.action_space.seed(self.config.seed)
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        
        Returns:
            (observation, info) 元组
        """
        reset_result = self.env.reset(seed=self.config.seed)
        if isinstance(reset_result, tuple):
            observation, info = reset_result[:2]
        else:
            observation = reset_result
            info = {}
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 动作索引
            
        Returns:
            (next_state, reward, terminated, truncated, info) 元组
        """
        return self.env.step(action)
    
    def close(self) -> None:
        """关闭环境"""
        self.env.close()


class PPOTrainer:
    """PPO训练器类"""
    
    def __init__(self, config: PPOConfig):
        """
        初始化训练器
        
        Args:
            config: PPO配置对象
        """
        self.config = config
        self.environment = CartPoleEnvironment(config)
        self.agent = PPOAgent(config, self.environment.state_dim, self.environment.action_dim)
        self.return_list: List[float] = []
    
    def train(self) -> List[float]:
        """
        训练智能体
        
        Returns:
            每个episode的回报列表
        """
        self.return_list = []
        
        print(f"开始训练PPO智能体...")
        print(f"环境: {self.config.env_name}")
        print(f"设备: {self.config.device}")
        print(f"训练轮数: {self.config.num_episodes}")
        print("-" * 50)
        
        for iteration in range(10):
            episodes_per_iteration = int(self.config.num_episodes / 10)
            
            with tqdm(total=episodes_per_iteration, desc=f'Iteration {iteration}') as pbar:
                for episode_idx in range(episodes_per_iteration):
                    episode_return = self._run_episode()
                    self.return_list.append(episode_return)
                    
                    # 更新进度条
                    if (episode_idx + 1) % 10 == 0:
                        current_episode = episodes_per_iteration * iteration + episode_idx + 1
                        avg_return = np.mean(self.return_list[-10:])
                        pbar.set_postfix({
                            'episode': f'{current_episode}',
                            'return': f'{avg_return:.3f}'
                        })
                    pbar.update(1)
        
        print("\n训练完成！")
        return self.return_list
    
    def _run_episode(self) -> float:
        """
        运行一个episode
        
        Returns:
            episode的累积回报
        """
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': []
        }
        
        state, _ = self.environment.reset()
        done = False
        episode_return = 0
        
        while not done:
            # 选择动作
            action = self.agent.take_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = self.environment.step(action)
            done = terminated or truncated
            
            # 存储转移
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            
            state = next_state
            episode_return += reward
        
        # 更新策略
        self.agent.update(transition_dict)
        
        return episode_return
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取训练统计信息
        
        Returns:
            包含统计信息的字典
        """
        if not self.return_list:
            return {}
        
        return {
            'mean_return': np.mean(self.return_list),
            'max_return': np.max(self.return_list),
            'min_return': np.min(self.return_list),
            'std_return': np.std(self.return_list),
            'last_10_mean': np.mean(self.return_list[-10:]),
            'last_10_std': np.std(self.return_list[-10:])
        }


class ResultVisualizer:
    """结果可视化类"""
    
    def __init__(self, config: PPOConfig):
        """
        初始化可视化器
        
        Args:
            config: PPO配置对象
        """
        self.config = config
        plt.rcParams['figure.figsize'] = (12, 5)
        plt.rcParams['font.size'] = 10
    
    def plot_results(self, return_list: List[float]) -> None:
        """
        绘制训练结果
        
        Args:
            return_list: 每个episode的回报列表
        """
        episodes_list = list(range(len(return_list)))
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 绘制原始回报曲线
        ax1.plot(episodes_list, return_list, alpha=0.6, color='blue', linewidth=1)
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Returns')
        ax1.set_title(f'PPO on {self.config.env_name} - Raw Returns')
        ax1.grid(True, alpha=0.3)
        
        # 绘制移动平均曲线
        mv_return = rl_utils.moving_average(return_list, 9)
        ax2.plot(episodes_list, mv_return, color='red', linewidth=2)
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Returns')
        ax2.set_title(f'PPO on {self.config.env_name} - Moving Average')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if self.config.save_plots:
            plt.savefig(self.config.plot_filename, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {self.config.plot_filename}")
        
        plt.show()
    
    def print_statistics(self, stats: Dict[str, float]) -> None:
        """
        打印统计信息
        
        Args:
            stats: 统计信息字典
        """
        print("\n" + "=" * 50)
        print("训练统计信息")
        print("=" * 50)
        print(f"平均回报:        {stats['mean_return']:.2f}")
        print(f"最大回报:        {stats['max_return']:.2f}")
        print(f"最小回报:        {stats['min_return']:.2f}")
        print(f"回报标准差:      {stats['std_return']:.2f}")
        print(f"最后10轮平均回报: {stats['last_10_mean']:.2f}")
        print(f"最后10轮标准差:  {stats['last_10_std']:.2f}")
        print("=" * 50)


def main():
    """主函数"""
    # 创建配置
    config = PPOConfig(
        env_name='CartPole-v1',
        seed=0,
        num_episodes=100,
        enable_visualization=False,
        save_plots=True
    )
    
    # 创建训练器
    trainer = PPOTrainer(config)
    
    # 训练
    return_list = trainer.train()
    
    # 获取统计信息
    stats = trainer.get_statistics()
    
    # 可视化结果
    visualizer = ResultVisualizer(config)
    visualizer.plot_results(return_list)
    visualizer.print_statistics(stats)
    
    # 关闭环境
    trainer.environment.close()
    
    return return_list, stats


if __name__ == "__main__":
    main()

