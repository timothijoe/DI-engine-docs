ICM
====

概述
--------

ICM (Intrinsic Curiosity Module) 首次在论文
`Curiosity-driven Exploration by Self-supervised Prediction <http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf>`__ 中提出,
用于研究在稀疏奖励中，如何让 agent 更有效地探索环境，学习技能。它定义的‘好奇心’作为内在的信号，分为两个部分：逆向模型 (reward module) 和前向模型 (forward module)。

**特征空间的描述** 直接把原始图像来作为表征存在着一些缺陷， 比如预测像素是一件很困难的事情，同时有些环境元素并不受agent控制或者影响，因此原始像素图像可能会变得很难预测。
为了解决这种缺陷，可以在这里使用一个嵌入层，把环境的特征表征程一个特征向量。对于一个特征表征，应该可以提取两种环境元素的信息，即可以被agent控制的环境元素，以及可以影响agent的元素；同时应该忽略一种环境元素的信息，即既不会被agent控制，又不会影响agent的元素。
对于后面的前向模型和逆向模型，我们都使用这个特征向量作为环境的表征。就说明对于那些可以被agent控制的环境元素的特征表征得越好。
**逆向模型** 核心思想在于通过当前状态和下一个时刻的状态的表征，估计出当前状态采用的动作值。对于当前动作估计的越准确，说明对于agent可以控制的环境元素的表征就越好。
**前向模型** 核心思想在于通过当前状态表征和当前动作，估计出下一个时刻的状态表征。 这个模型可以让学到的状态表征更加容易预测。

ICM的agent有两个子系统： 一个子系统是内在奖励生成器，它把前向模型和逆向模型的预测误差作为内在奖励（因此总奖励为内在奖励和稀疏的环境奖励之和）； 另一个子系统是一个策略网络，用于输出一系列的动作。训练策略网络的优化目标就是总分数的期望，因此策略的优化既会考虑让稀疏的环境奖励得到更多，也会探索此前没有见过的动作，以求得到更多的内在奖励。

核心要点
-----------

1. ICM的基线强化学习算法是 `A3C <http://proceedings.mlr.press/v48/mniha16.pdf>`__ ,可以参考我们的实现 `A2C <hhttps://github.com/opendilab/DI-engine/blob/main/ding/policy/a2c.py>`__ ，如果想实现A3C，可以使用多个环境同时训练。

2. 在后续的工作中 `Large-Scale Study of Curiosity-Driven Learning <https://arxiv.org/pdf/1808.04355v1.pdf>`__, 使用的基线算法是PPO， 可以参考我们的实现 `PPO <hhttps://github.com/opendilab/DI-engine/blob/main/ding/policy/ppo.py>`__，通过PPO算法，只需要少量的超参数微调，就可以获得鲁棒的学习效果。


3. 奖励归一化。 由于奖励是不稳定的，因此需要把奖励归一化到[0,1]之间，让学习更稳定。

4. 更多的actor（在DI-engine中为更多的collector)： 增加更多并行的actor可以使得训练更加稳定。

5. 特征归一化。 通过整合内在奖励和外在奖励的过程中，确保内在奖励在不同特征表述的缩放很重要，这一点可以通过batch normalization来实现。

关键方程或关键框图
---------------------------
NGU算法的整体训练与计算流程如下：

.. image:: images/ICM_illustraion.png
   :align: center
   :scale: 70%

1. 如左图所示，agent在状态 :math:`s_t` 通过当前的策略 :math:`\pi` 采样得到动作 a并执行，最终得到状态 :math:`s_{t+1}`。 总的奖励为两个部分奖励之和，一部分是外部奖励 :math:`r_t^e`，即环境中得到的，稀疏的奖励；另一部分是由ICM得到的内在奖励 :math:`r_t^ｉ` （具体计算过程由第4步给出），最终的策略的需要通过优化总的奖励来实现训练的目的。
具体公式表现为：

 :math:`ｒ_t=r_t^i + r_t^e` 

 :math:`{\max}_{\theta_p}\mathbb{E}_{\pi(s_t;\theta_p)}[\Sigma_t r_t]`

2. 在ICM的逆向模型中，它首先会把　:math:`s_t`　和 :math:`s_{t+1}`　提取表征后的特征向量　:math:`\Phi(s_t; \theta_E)`　和　:math:`\Phi(s_{t+1}; \theta_E)` 作为输入（后面把它们简化为 :math:`\Phi(s_t)` 和 :math:`\Phi(s_{t+1})`），并且输出预测的动作值　:math:`a_t` 。

 :math:`\hat{a_t}=g(\Phi(s_t),\Phi(s_{t+1}) ; \theta_I)` 

 :math:`{\min}_{\theta_I, \theta_E} L_i(\hat{a_t},a_t)` 

在这里　:math:`\hat{a_t}`　是　:math:`a_t`　的预测值， :math:`L_Ｉ` 描述两者之间的差异 (softmax loss) 。差异越小，说明对于当前动作估计的越准确，说明对于agent可以控制的环境元素的表征就越好。


３．ICM的前向模型会把　:math:`\Phi(s_t)`　和动作值　:math:`a_t`　作为输入，　输出下一个时刻状态的特征向量的预测数值　:math:`\hat{\Phi}(s_{t+1})` 。
下一时刻预测的特征向量和真实的特征向量的误差被用来当做内在奖励。

 :math:`\hat{\phi(s_{t+1})}=f(\Phi(s_t),a_t) ; \theta_F)` 

 :math:`{\min}_{\theta_F, \theta_E} L_F(\hat{\phi(s_{t+1})},\phi(s_{t+1}))`

在这里， :math:`L_Ｆ` 描述了　:math:`\hat{\phi(s_{t+1})}` 和 :math:`\phi(s_{t+1})` 之间的差异 (regression loss), 通过前向模型的学习，可以让学习的特征表征更加容易预测。


４．内在奖励可以由　:math:`\hat{\phi(s_{t+1})}` 和 :math:`\phi(s_{t+1})` 之间的差异来表征：

 :math:`r_i^t = \frac{\eta}{2} (\| \hat{\phi(s_{t+1})} - \phi(s_{t+1}) \|)_2^2` 

**总结**：
ICM通过前向模型和逆向模型，会提取更多会受到agent影响的环境元素特征；对于那些agent的动作无法影响的环境元素（比如噪声），将不会产生内在奖励，进而提高了探索策略的鲁棒性。
同时，１－４也可以写作一个优化函数：

 :math:`{\min}_{\theta_P,\theta_I,\theta_F，\theta_E} [- \lambda \mathbb{E}_{\pi(s_t;\theta_p)}[\Sigma_t r_t] + (1-\beta)L_I + \beta LF]`
在这里 :math:`\beta \in [0,1]` 用来权衡正向模型误差和逆向模型误差的权重；　:math:`\lambda >0` 用来表征策略梯度误差对于内在信号的重要程度。


重要实现细节
-----------
1. 奖励归一化。在通过上面所述的算法计算得到局内内在奖励后，由于在智能体学习的不同阶段和不同的环境下，它的幅度是变化剧烈的，如果直接用作后续的计算，很容易造成学习的不稳定。在我们
的实现中，是按照下面的最大最小归一化公式 归一化到[0,1]之间:
``episodic_reward = (episodic_reward - episodic_reward.min()) / (episodic_reward.max() - episodic_reward.min() + 1e-11)``，
其中episodic_reward是一个mini-batch计算得到的局内内在奖励。我们也分析了其他归一化方式的效果。

    方法1: transform to batch mean1: erbm1
    由于我们的实现中批数据里面可能会有null_padding的样本(注意null_padding样本的原始归一化前的episodic reward=0)，造成episodic_reward.mean()不是真正的均值，需要特别处理计算得到真实的均值episodic_reward_real_mean，
    这给代码实现造成了额外的复杂度，此外这种方式不能将局内内在奖励的幅度限制在一定范围内，造成内在奖励的加权系数不好确定。
    ``episodic_reward = episodic_reward / (episodic_reward.mean() + 1e-11)``

    方法2. transform to long-term mean1: erlm1
    存在和方法1类似的问题
    ``episodic_reward = episodic_reward / self._running_mean_std_episodic_reward.mean``

    方法3. transform to mean 0, std 1
    由于rnd_reward在[1,5]集合内, episodic reward 应该大于0,例如如果episodic_reward是 -2, rnd_reward 越大, 总的intrinsic reward却越小, 这是不正确的
    ``episodic_reward = (episodic_reward - self._running_mean_std_episodic_reward.mean)/ self._running_mean_std_episodic_reward.std``

    方法4. transform to std1, 似乎没有直观的意义
    ``episodic_reward = episodic_reward / self._running_mean_std_episodic_reward.std``

2. 在minigrid环境上，由于环境设置只有在智能体达到目标位置时，智能体才获得一个正的0到1之间的奖励，其他时刻奖励都为零，在这种环境上累计折扣内在奖励的幅度会远大于原始的0，1之间的数，造成
智能体学习的目标偏差太大，为了缓解这个问题，我们在实现中对每一局的最后一个非零的奖励乘上一个权重因子，实验证明如果不加这个权重因子，在最简单的empty8环境上算法也不能收敛，这显示了原始
外在奖励和内在奖励之间相对权重的重要性。


实现
---------------
内在好奇心模型( ``ICMRewardModel`` )的接口定义如下：

.. autoclass:: ding.reward_model.ngu_reward_model.RndNGURewardModel
   :members: __init__, estimate
   :noindex:

ICMNetwork
~~~~~~~~~~~~~~~~~
首先我们定义类　``ICMNetwork`` 涉及四种神经网络：

self.feature: 对observation的特征进行提取；

self.inverse_net: ICM网络的逆向模型，通过将先后两帧feature特征作为输入，输出一个预测的动作

self.residual: 参与ICM网络的前向模型，通过多次将action与中间层的输出做concat,使得特征更加明显

self.forward_net: 参与ICM网络的前向模型，负责输出 :math:`s_{t+1}`时刻的feature

        .. code-block:: python

            class ICMNetwork(nn.Module):

                def __init__(self, obs_shape: Union[int, SequenceType], hidden_size_list: SequenceType, action_shape: int) -> None:
                    super(ICMNetwork, self).__init__()
                    if isinstance(obs_shape, int) or len(obs_shape) == 1:
                        self.feature = FCEncoder(obs_shape, hidden_size_list)
                    elif len(obs_shape) == 3:
                        self.feature = ConvEncoder(obs_shape, hidden_size_list)
                    else:
                        raise KeyError(
                            "not support obs_shape for pre-defined encoder: {}, please customize your own ICM model".
                            format(obs_shape)
                        )
                    self.action_shape = action_shape
                    feature_output = hidden_size_list[-1]
                    self.inverse_net = nn.Sequential(nn.Linear(feature_output * 2, 512), nn.ReLU(), nn.Linear(512, action_shape))
                    self.residual = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(action_shape + 512, 512),
                            nn.LeakyReLU(),
                            nn.Linear(512, 512),
                        ) for _ in range(8)
                    ])
                    self.forward_net_1 = nn.Sequential(nn.Linear(action_shape + feature_output, 512), nn.LeakyReLU())
                    self.forward_net_2 = nn.Sequential(nn.Linear(action_shape + 512, feature_output), )

                def forward(self, state: torch.Tensor, next_state: torch.Tensor,
                            action_long: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    action = one_hot(action_long, num=self.action_shape).squeeze(1)
                    encode_state = self.feature(state)
                    encode_next_state = self.feature(next_state)
                    # get pred action logit
                    pred_action_logit = torch.cat((encode_state, encode_next_state), 1)
                    pred_action_logit = self.inverse_net(pred_action_logit)
                    # ---------------------

                    # get pred next state
                    pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
                    pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

                    # residual
                    for i in range(4):
                        pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
                        pred_next_state_feature_orig = self.residual[i * 2 + 1](
                            torch.cat((pred_next_state_feature, action), 1)
                        ) + pred_next_state_feature_orig
                    pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))
                    real_next_state_feature = encode_next_state
                    return real_next_state_feature, pred_next_state_feature, pred_action_logit






参考资料
---------
1. Pathak D, Agrawal P, Efros A A, et al. Curiosity-driven exploration by self-supervised prediction[C]//International conference on machine learning. PMLR, 2017: 2778-2787.

2. Burda Y, Edwards H, Storkey A, et al. Exploration by random network distillation[J]. https://arxiv.org/abs/1810.12894v1. arXiv:1810.12894, 2018.
