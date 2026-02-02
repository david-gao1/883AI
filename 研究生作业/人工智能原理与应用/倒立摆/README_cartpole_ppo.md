# CartPole PPO - 面向对象版本

## 文件说明

`cartpole_ppo.py` 是将原始 Jupyter Notebook 代码重构为面向对象编程的 Python 文件，提高了代码的可读性、可维护性和可复用性。

## 代码结构

### 1. PPOConfig (配置类)
- 使用 `@dataclass` 装饰器，集中管理所有超参数
- 包含环境配置、网络配置、训练配置、PPO算法参数等
- 自动检测并设置计算设备（CPU/GPU）

### 2. PolicyNet (策略网络)
- 继承自 `torch.nn.Module`
- 输出动作概率分布
- 用于 Actor（策略网络）

### 3. ValueNet (价值网络)
- 继承自 `torch.nn.Module`
- 估计状态价值函数
- 用于 Critic（价值网络）

### 4. PPOAgent (PPO智能体)
- 封装策略网络和价值网络
- 实现动作选择（`take_action`）
- 实现PPO算法更新（`update`）

### 5. CartPoleEnvironment (环境封装)
- 封装 Gymnasium 环境
- 处理环境初始化和种子设置
- 提供统一的环境接口

### 6. PPOTrainer (训练器)
- 管理训练流程
- 运行episode并收集数据
- 提供训练统计信息

### 7. ResultVisualizer (结果可视化)
- 绘制训练曲线
- 打印统计信息
- 保存训练结果图片

## 使用方法

### 基本使用

```python
from cartpole_ppo import PPOConfig, PPOTrainer, ResultVisualizer

# 创建配置
config = PPOConfig(
    num_episodes=100,
    enable_visualization=False
)

# 创建训练器并训练
trainer = PPOTrainer(config)
return_list = trainer.train()

# 可视化结果
visualizer = ResultVisualizer(config)
visualizer.plot_results(return_list)
stats = trainer.get_statistics()
visualizer.print_statistics(stats)
```

### 直接运行

```bash
python cartpole_ppo.py
```

### 自定义配置

```python
from cartpole_ppo import PPOConfig, PPOTrainer, ResultVisualizer

# 自定义配置
config = PPOConfig(
    env_name='CartPole-v1',
    seed=42,
    num_episodes=200,
    hidden_dim=256,
    actor_lr=5e-4,
    critic_lr=5e-3,
    gamma=0.99,
    lmbda=0.95,
    epochs=10,
    eps=0.2,
    enable_visualization=False,
    save_plots=True,
    plot_filename='my_training_results.png'
)

# 训练
trainer = PPOTrainer(config)
return_list = trainer.train()

# 查看结果
visualizer = ResultVisualizer(config)
visualizer.plot_results(return_list)
stats = trainer.get_statistics()
visualizer.print_statistics(stats)
```

## 依赖库

```bash
pip install torch gymnasium matplotlib numpy tqdm
```

## 代码优势

### 1. 面向对象设计
- 每个类职责单一，易于理解和维护
- 代码结构清晰，符合SOLID原则

### 2. 配置集中管理
- 所有超参数集中在 `PPOConfig` 类中
- 易于调整和实验不同参数组合

### 3. 类型提示
- 使用类型提示提高代码可读性
- IDE可以更好地提供代码补全和错误检查

### 4. 模块化设计
- 各个组件独立，可以单独测试和复用
- 易于扩展新功能

### 5. 文档完善
- 每个类和方法都有详细的文档字符串
- 参数和返回值都有说明

## 与原始代码的对比

| 特性 | 原始Notebook | 面向对象版本 |
|------|-------------|-------------|
| 代码组织 | 过程式，分散在多个cell | 面向对象，类封装 |
| 配置管理 | 分散的变量 | 集中的Config类 |
| 可复用性 | 低 | 高 |
| 可测试性 | 低 | 高 |
| 可维护性 | 中等 | 高 |
| 类型安全 | 无 | 有类型提示 |

## 扩展建议

1. **添加日志记录**：使用 `logging` 模块记录训练过程
2. **模型保存/加载**：添加保存和加载模型的功能
3. **早停机制**：当性能不再提升时自动停止训练
4. **超参数搜索**：集成超参数优化工具
5. **多环境支持**：支持训练其他Gymnasium环境
6. **分布式训练**：支持多GPU训练

## 示例输出

```
开始训练PPO智能体...
环境: CartPole-v1
设备: cpu
训练轮数: 100
--------------------------------------------------
Iteration 0: 100%|██████████| 10/10 [00:02<00:00,  4.21it/s, episode=10, return=220.300]
...
训练完成！

训练曲线已保存到: ppo_training_results.png

==================================================
训练统计信息
==================================================
平均回报:        380.45
最大回报:        500.00
最小回报:        125.00
回报标准差:      95.32
最后10轮平均回报: 450.20
最后10轮标准差:  35.15
==================================================
```

