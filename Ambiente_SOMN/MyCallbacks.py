import os
import csv
from cv2 import Algorithm
import matplotlib.pyplot as plt
from datetime import date, datetime
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
import wandb

class MyCallbacks(DefaultCallbacks):

    # Obter a hora atual
    hora_atual = datetime.now()

    # Formatando a hora como uma string
    hora_formatada = hora_atual.strftime("%Y-%m-%d/%H-%M-%S")

    def __init__(self, save_dir=f"plots/{hora_formatada}"):
        super().__init__()

        os.makedirs(save_dir, exist_ok=True)

        self.save_dir = save_dir
        self.csv_path_yard = os.path.join(save_dir, "Yard.csv")
        self.csv_path_action = os.path.join(save_dir, "Action.csv")
        self.csv_path_reject = os.path.join(save_dir, "Reject.csv")
        self.csv_path_rejectwwast = os.path.join(save_dir, "Reject_w_wast.csv")
        self.csv_path_rejectall = os.path.join(save_dir, "RejectAll.csv")
        self.csv_path_aceptreject = os.path.join(save_dir, "AceptReject.csv")

        self.step = 0
        self.numepisode = 0
        self.wandb_initialized = False
        
    def on_episode_start(self, *, worker: RolloutWorker, base_env, policies, episode, env_index, **kwargs) -> None:
        self.yard_hist = {}
        self.action_hist = {}
        self.reject = {}
        self.reject_w = {}
        self.reject_all = 0
        self.acept = {}
        self.sum_acept = 0
        self.sum_reject = 0
        self.acept_produced = {}
        self.acept_yard = {}
        self.total_match = {}
        self.match_reject = {}
        self.penalty = {}
        self.penaltyAll = 0
        self.wastAll = 0
        self.matchRejectYardAll = 0
        self.produced_yard = {}
        self.produced_yard_All = 0
        self.prouced_wast = {}
        self.prouced_wast_All = 0

    def on_episode_step(self, *, worker: RolloutWorker, base_env, policies=None, episode, env_index=None, **kwargs):
        self.step+=1
        for agent_id in episode.get_agents():
            agent_info = episode.last_info_for(agent_id)
            
            yard_value = agent_info.get("yard")
            action_value = agent_info.get("acoes")
            reject_value = agent_info.get("reject")
            rejectw_value = agent_info.get("reject_w_west")
            acept_value = agent_info.get("acept_reject")
            produced_yard_value = agent_info.get("yard_reject")
            produced_wast_value = agent_info.get("produced_wast")

            self.produced_yard[agent_id] = produced_yard_value
            self.prouced_wast[agent_id] = produced_wast_value

            self.reject[agent_id] = reject_value
            self.reject_w[agent_id] = rejectw_value

            self.yard_hist[agent_id] = yard_value
            self.action_hist[agent_id] = action_value

            self.acept[agent_id] = acept_value

            self.acept_produced[agent_id] = agent_info.get("produced_reject")
            self.acept_yard[agent_id] = agent_info.get("yard_reject")
            self.total_match[agent_id] = agent_info.get("total_match")
            self.match_reject[agent_id] = agent_info.get("Math_Reject")
            self.penalty[agent_id] = agent_info.get("penalty")

            episode.custom_metrics[f"Yard agent {agent_id}"] = yard_value
            episode.custom_metrics[f"Action agent {agent_id}"] = action_value
            episode.custom_metrics[f"Acept_reject agent {agent_id}"] = acept_value
            episode.custom_metrics[f"Penalty Agent {agent_id}"] = self.penalty[agent_id]
            
            episode.custom_metrics[f"environmental profit"] = agent_info.get("LU")
            episode.custom_metrics[f"environmental variability"] = agent_info.get("VA")
            episode.custom_metrics[f"environmental sustainability"] = agent_info.get("SU")
            
            episode.custom_metrics[f"enviromental max profit"] = agent_info.get("max_LU")
            episode.custom_metrics[f"enviromental max variability"] = agent_info.get("max_VA")
            episode.custom_metrics[f"enviromental max sustainability"] = agent_info.get("max_SU")
            
            
            self.penaltyAll += self.penalty[agent_id]

            # Verifica se o valor de "yard" está presente
            # if yard_value is not None:
            #     # Salva os dados em CSV
            #     self.salvar_dados_csv_yard(self.step, agent_id, yard_value)
            
            # if action_value is not None:
            #     self.salvar_dados_csv_action(self.step, agent_id, action_value)
        
            self.reject_all = agent_info.get("reject_all")
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env, policies, episode, env_index, **kwargs) -> None:
        episode.custom_metrics[f"Reject_All"] = self.reject_all
        for agent_id in self.yard_hist:
            episode.custom_metrics[f"reject agent {agent_id}"] = self.reject[agent_id]
            episode.custom_metrics[f"Reject_w_wast {agent_id}"] = self.reject_w[agent_id]
            episode.custom_metrics[f"Acept_produced_agent {agent_id}"] = self.acept_produced[agent_id]
            episode.custom_metrics[f"Acept_Yard_agent {agent_id}"] = self.acept_yard[agent_id]
            episode.custom_metrics[f"Total_Match_agent {agent_id}"] = self.total_match[agent_id]
            episode.custom_metrics[f"Match_Reject_agent {agent_id}"] = self.match_reject[agent_id]

            self.wastAll += self.reject_w[agent_id]

            self.numepisode += 1
            self.sum_acept += self.acept[agent_id]
            self.sum_reject += self.reject[agent_id]

            self.produced_yard_All += self.produced_yard[agent_id]
            self.prouced_wast_All += self.prouced_wast[agent_id]

            # self.salvar_dados_csv_reject(self.numepisode, agent_id, self.reject[agent_id])
            # self.salvar_dados_csv_reject_w_wast(self.numepisode, agent_id, self.reject_w[agent_id])
            # self.salvar_dados_csv_acept(self.numepisode, agent_id, self.acept[agent_id])

        # self.salvar_dados_csv_reject_all(self.numepisode, self.reject_all)
        episode.custom_metrics[f"Sum_Acept"] = self.sum_acept
        episode.custom_metrics[f"Sum_Reject"] = self.sum_reject
        episode.custom_metrics[f"Penalty All"] = self.penaltyAll
        episode.custom_metrics[f"Reject_w_wast_All"] = self.wastAll
        episode.custom_metrics[f"Produced_Yard_All"] = self.produced_yard_All
        episode.custom_metrics[f"Produced_Wast_All"] = self.prouced_wast_All