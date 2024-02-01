from typing import Dict, Optional, Union
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

class MyCallbacks(DefaultCallbacks):

    def on_episode_step(self, *, worker: RolloutWorker, base_env, policies=None, episode, env_index=None, **kwargs):
        for agent_id in episode.get_agents():
            agent_info = episode.last_info_for(agent_id)
            # Registre as recompensas e informações adicionais
            # episode.custom_metrics[f"Reward_lastfunction_agent_{agent_id}"] = episode.last_reward_for(agent_id)
            # episode.custom_metrics[f"Last_Action_agent_{agent_id}"] = episode.last_action_for(agent_id)
            episode.custom_metrics[f"Reward_agente_{agent_id}"] = agent_info["rw"]
            episode.custom_metrics[f"Reward_agente_pr_{agent_id}"] = agent_info["rw_pr"]
            episode.custom_metrics[f"Reward_agente_va_{agent_id}"] = agent_info["rw_va"]
            episode.custom_metrics[f"Reward_agente_su_{agent_id}"] = agent_info["rw_su"]

            

        # Certifique-se de chamar a implementação da classe pai
        # return super().on_episode_step(worker=worker,base_env=base_env,policies=policies, episode=episode,env_index=env_index,**kwargs)
