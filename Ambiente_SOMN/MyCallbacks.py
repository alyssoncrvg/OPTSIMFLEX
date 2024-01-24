from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker

class MyCallbacks(DefaultCallbacks):
    def on_episode_step(self, *, worker: RolloutWorker, base_env, policies=None, episode: Episode, env_index=None, **kwargs):
        for agent_id in episode.get_agents():
            agent_info = episode.last_info_for(agent_id)

            episode.custom_metrics[f"Reward Agente {agent_id}"] = agent_info["rw"]
            episode.custom_metrics[f"Reward Lucro Agente {agent_id}"] = agent_info["rw_pr"]
            episode.custom_metrics[f"Reward Sustentabilidade Agente {agent_id}"] = agent_info["rw_su"]
            episode.custom_metrics[f"Reward Variabilidade Agente {agent_id}"] = agent_info["rw_va"]
            # episode.custom_metrics[f"Variabilidade Agente {agent_id}"] = agent_info["VA"]
            # episode.custom_metrics[f"Sustentabilidade Agente {agent_id}"] = agent_info["SU"]
            # episode.custom_metrics[f"Ações Agente {agent_id}"] = agent_info["acoes"]
            # episode.custom_metrics[f"Atrasos Reais Agente {agent_id}"] = agent_info["atrasos_reais"]
            # episode.custom_metrics[f"acao_on_state_plan Agente {agent_id}"] = agent_info["acao_on_state_plan"]
            # episode.custom_metrics[f"carga_on_state_plan Agente {agent_id}"] = agent_info["carga_on_state_plan"]
            # episode.custom_metrics[f"patio_on_state_plan Agente {agent_id}"] = agent_info["patio_on_state_plan"]