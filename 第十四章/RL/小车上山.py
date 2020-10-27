import gym
env=gym.make('MountainCar-v0')
print('观测空间={}'.format(env.observation_space))
print('动作空间={}'.format(env.action_space))
print('观测范围={}~{}'.format(env.observation_space.low,env.observation_space.high))
print('动作数={}'.format(env.action_space.n))

#根据指定确定性策略决定动作的智能体
class BespokeAgent:
    def __init__(self,env):
        pass

    def decide(self,observation):    #决策
        position,velocity=observation
        lb=min(-0.09*(position+0.25)**2+0.03,
               0.3*(position+0.9)**4-0.008)

