from typing import Any, ClassVar, Optional, TypeVar, Union

import torch as th
from gymnasium import spaces
    # 定义强化学习环境的动作和观察空间。spaces.Box：连续空间；spaces.Discrete：离散空间；spaces.MultiDiscrete：多维离散空间；spaces.Dict：字典空间（多输入环境）
from torch.nn import functional as F
    # 提供神经网络的函数式接口和激活函数。F.relu()：ReLU激活函数；F.softmax()：Softmax函数（用于策略输出）；F.mse_loss()：均方误差损失；F.cross_entropy()：交叉熵损失

from stable_baselines3.common.buffers import RolloutBuffer
    # 经验缓冲回放区，存储与环境交互的经验（状态、动作、奖励等）。收集一个回合（rollout）的数据；计算优势估计和回报；批量采样用于训练
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
    # 所有同策略算法（如A2C、PPO）的基类。通用的训练循环；与环境交互的逻辑；模型保存/加载方法；基本的超参数
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
    # 策略网络。ActorCriticPolicy：标准MLP，处理向量状态；ActorCriticCnnPolicy：CNN网络，处理图像状态；MultiInputActorCriticPolicy：多模态输入（如向量+图像）
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
    # 定义常用的类型别名，统一接口。GymEnv：强化学习环境接口；MaybeCallback：可选的回调函数；Schedule：学习率/熵系数等参数的调度函数
from stable_baselines3.common.utils import explained_variance
    # 计算优势函数的解释方差。评估价值函数预测的好坏；监控训练过程；公式：1 - Var(A - V)/Var(A)，越接近1越好

SelfA2C = TypeVar("SelfA2C", bound="A2C")


class A2C(OnPolicyAlgorithm):
    """
    Mark Point
    
    Advantage Actor Critic (A2C)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
        # 学习率
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
        # 折扣因子
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
        # 广义优势估计函数因子
    :param ent_coef: Entropy coefficient for the loss calculation
        # 熵系数，策略梯度算法中的正则化项系数，用于鼓励探索
    :param vf_coef: Value function coefficient for the loss calculation
        # 价值函数损失系数，用于平衡策略损失和价值损失在总损失中的权重
    :param max_grad_norm: The maximum value for the gradient clipping
        # 梯度裁剪是一种稳定深度学习训练的技术，通过限制梯度的大小来防止梯度爆炸问题
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
        # RMSProp优化器的epsilon参数，用于数值稳定性的一个小常数，RMSProp（Root Mean Square Propagation）是一种自适应学习率优化算法，特别适合处理非平稳目标（如强化学习）。
        # 两者都用于稳定训练，但方式不同：
        # rms_prop_eps: 自适应学习率中的数值稳定性
        # max_grad_norm: 硬性梯度裁剪
        
        # 通常一起使用：
        # optimizer = RMSprop(eps=rms_prop_eps)  # 自适应调整学习率
        # torch.nn.utils.clip_grad_norm_(..., max_norm=max_grad_norm)  # 硬性限制
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
        # 优化器选择参数，用于决定A2C算法使用RMSprop还是Adam优化器
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
        # 广义状态依赖探索（State-Dependent Exploration, gSDE）选择参数，用于替代传统动作噪声的现代探索策略，是一种自适应的探索策略
        # 传统动作噪声 vs gSDE
        # 传统方法：动作 = 策略输出 + 固定噪声（如高斯噪声）
        # gSDE方法：动作 = 策略输出 + 状态依赖的自适应噪声
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
        # gSDE噪声采样频率参数，用于控制状态依赖探索噪声的更新频率
        # 默认-1：每个rollout/episode采样一次（最稳定）
        # n>0：每n步采样一次（平衡探索与稳定性）
        # 1：每步采样（最大探索）
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
        # Rollout缓冲区类选择参数，用于自定义经验回放缓冲区，是存储智能体与环境交互经验的容器，在同策略算法（如A2C、PPO）中特别重要。
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
        # Rollout缓冲区创建参数，用于向自定义缓冲区传递额外参数
    :param normalize_advantage: Whether to normalize or not the advantage
        # 优势归一化参数，用于标准化优势函数的尺度
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
        # 统计窗口大小参数，用于平滑训练日志的滚动平均计算，这个参数不直接影响算法性能，但极大影响训练监控和超参数调优的效率
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
        # TensorBoard日志目录参数，用于启用和配置TensorBoard训练可视化
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`a2c_policies`
        # 策略网络参数，用于自定义策略网络架构和配置
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
        # 详细输出控制参数，用于控制训练过程中的信息输出级别
    :param seed: Seed for the pseudo random generators
        # 随机种子参数，用于控制实验的可重复性
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
        # 选择GPU或CPU进行计算
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
        # 模型初始化控制参数，用于延迟或控制神经网络构建时机
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()
                #                         梯度流向分析:
                # ============================================================
                # 策略网络（Actor）梯度:
                #   policy_loss = -(advantages * log_prob).mean()
                #   ∇policy_loss = -advantages * ∇log_prob / n
                #   梯度流向: log_prob → 策略网络参数
                #   优化方向:
                #     - advantages > 0: 增加该动作的概率
                #     - advantages < 0: 减少该动作的概率

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)
                #                         梯度流向分析:
                # ============================================================
                # 价值网络（Critic）梯度:
                #   value_loss = MSE(returns, values)
                #   ∇value_loss = 2*(values - returns) * ∇values / n
                #   梯度流向: values → 价值网络参数
                #   优化方向: 使 values 接近 returns
            
                # 整体优化目标:
                #   1. Actor: 选择高优势的动作
                #   2. Critic: 准确预测状态价值
                #   3. 两者协同: Critic为Actor提供准确的优势估计
            
            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                # 这里policy_loss和entropy_loss涉及到policy_net，两者在梯度前向传播和反向传播时，是有所区别的。
                # 它们更新的是完全相同的参数集，但通过不同的梯度方向进行更新。
                # 梯度流可视化：
                    # 策略网络参数 θ = {θ_feature, θ_mean, θ_log_std}
    
                    # 正向传播：
                    #     特征 h = f(obs; θ_feature)
                    #     均值 μ = g_mean(h; θ_mean)
                    #     对数标准差 log_std = g_std(h; θ_log_std)
                    #     标准差 σ = exp(log_std)
                    
                    # 计算：
                    #     log_prob = -0.5*((a-μ)/σ)² - log_std - 0.5*log(2π)
                    #     entropy = 0.5 + log(√(2π)) + log_std
                    
                    # 反向传播：
                    # 对于 log_prob 的梯度：
                    #     ∇_θ log_prob = 
                    #         [∂log_prob/∂μ] * [∂μ/∂θ] +               ← 通过 μ 的路径
                    #         [∂log_prob/∂σ] * [∂σ/∂log_std] * [∂log_std/∂θ]   ← 通过 σ 的路径
                            
                    #     其中：
                    #         ∂log_prob/∂μ = (a-μ)/σ²
                    #         ∂log_prob/∂σ = ((a-μ)²/σ³) - (1/σ)
                    #         ∂σ/∂log_std = σ = exp(log_std)
                            
                    #     简化后：
                    #         ∂log_prob/∂log_std = ((a-μ)²/σ² - 1)  ← 注意这个简化！
                                    # 代入 σ = exp(log_std)：
                                    # log_prob = -0.5*(a-μ)² * exp(-2*log_std) - log_std - 0.5*log(2π)
                                    # 直接对 log_std 求导：
                                    # ∂log_prob/∂log_std = (a-μ)² * exp(-2*log_std) - 1
                                    #                    = (a-μ)²/σ² - 1
                                    # 这与链式法则结果一致：
                                    # ∂log_prob/∂log_std = ∂log_prob/∂σ * ∂σ/∂log_std
                                    #                    = [((a-μ)²/σ³ - 1/σ)] * σ
                                    #                    = (a-μ)²/σ² - 1
                    # 对于 entropy 的梯度：
                    #     ∇_θ entropy = 
                    #         [∂entropy/∂log_std] * [∂log_std/∂θ]   ← 直接路径

                    # 
                    # 初始策略分布：
                    #     π(a|s) ~ N(μ, σ)  # 有一定随机性
                    
                    # 策略损失的作用：
                    #     A>0的动作：概率密度↑
                    #     A<0的动作：概率密度↓
                    #     结果：分布变尖峰 → 确定性↑ → 熵↓
                    
                    # 熵损失的作用：
                    #     增加 σ → 分布变平坦 → 随机性↑ → 熵↑
                    # 更新参数：
                    #     shared.weight: 两种损失都有梯度
                    #     shared.bias: 两种损失都有梯度  
                    #     mean.weight: 只有policy_loss有梯度
                    #     mean.bias: 只有policy_loss有梯度
                    #     log_std: 两种损失都有梯度，但方向可能不同

            # Optimization step
            self.policy.optimizer.zero_grad()
                # 作用：将模型中所有参数的梯度缓冲区清零
                # 原因：PyTorch的梯度是累加的。如果不清零，新计算的梯度会加到旧的梯度上，导致梯度爆炸和不正确的更新
            loss.backward()
                # 作用：自动微分，计算损失函数对模型所有可训练参数的梯度
                # 反向传播的数学原理：
                # 1. 从loss开始，沿着计算图向后传播
                # 2. 对每个参数计算 ∂loss/∂param
                # 3. 将每个参数的梯度存储在 param.grad 属性中

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # 作用：梯度裁剪，防止梯度爆炸
                # 算法原理：
                # 1. 计算所有参数的梯度向量的L2范数（欧几里得长度）
                # 2. 如果范数 > max_grad_norm，将所有梯度按比例缩放
            self.policy.optimizer.step()
                # 作用：根据计算出的梯度更新模型参数

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self: SelfA2C,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "A2C",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfA2C:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
