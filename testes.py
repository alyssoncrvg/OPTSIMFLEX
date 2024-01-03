from ray.rllib.env.multi_agent_env import make_multi_agent
from Ambiente_SOMN.make_env import make_env
from ray.rllib.algorithms.ppo import PPOConfig


ma_stateless_cartpole_cls = make_multi_agent(
    lambda config: make_env(0,0))
# Create a 3 agent multi-agent stateless cartpole.
ma_stateless_cartpole = ma_stateless_cartpole_cls(
    {"num_agents": 3})

obs = ma_stateless_cartpole.reset()
print(obs)

config = PPOConfig().environment(env=ma_stateless_cartpole).training(train_batch_size=4000)

algo = config.build()

print(algo.train())