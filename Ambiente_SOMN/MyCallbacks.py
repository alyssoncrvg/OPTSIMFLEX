import os
import csv
import matplotlib.pyplot as plt
from datetime import date, datetime
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker

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

        # Abre o arquivo CSV para escrita no modo 'a' (append)
        with open(self.csv_path_yard, mode='a', newline='') as arquivo_csv:
            escritor_csv_yard = csv.writer(arquivo_csv)

            # Escreve o cabeçalho se o arquivo estiver vazio
            if os.path.getsize(self.csv_path_yard) == 0:
                escritor_csv_yard.writerow(["Step", "Agent_ID", "Yard"])
        
        with open(self.csv_path_action, mode='a', newline='') as arquivo_csv:
            escritor_csv_action = csv.writer(arquivo_csv)

            # Escreve o cabeçalho se o arquivo estiver vazio
            if os.path.getsize(self.csv_path_action) == 0:
                escritor_csv_action.writerow(["Step", "Agent_ID", "Action"])
        
        with open(self.csv_path_reject, mode='a', newline='') as arquivo_csv:
            escritor_csv_action = csv.writer(arquivo_csv)

            # Escreve o cabeçalho se o arquivo estiver vazio
            if os.path.getsize(self.csv_path_action) == 0:
                escritor_csv_action.writerow(["Step", "Agent_ID", "Reject"])
        
        with open(self.csv_path_rejectwwast, mode='a', newline='') as arquivo_csv:
            escritor_csv_action = csv.writer(arquivo_csv)

            # Escreve o cabeçalho se o arquivo estiver vazio
            if os.path.getsize(self.csv_path_action) == 0:
                escritor_csv_action.writerow(["Step", "Agent_ID", "Reject_Wast"])
        
        with open(self.csv_path_aceptreject, mode='a', newline='') as arquivo_csv:
            escritor_csv_action = csv.writer(arquivo_csv)

            # Escreve o cabeçalho se o arquivo estiver vazio
            if os.path.getsize(self.csv_path_action) == 0:
                escritor_csv_action.writerow(["Step", "Agent_ID", "Acept_Reject"])
        
        with open(self.csv_path_rejectall, mode='a', newline='') as arquivo_csv:
            escritor_csv_action = csv.writer(arquivo_csv)

            # Escreve o cabeçalho se o arquivo estiver vazio
            if os.path.getsize(self.csv_path_action) == 0:
                escritor_csv_action.writerow(["Episode", "RejectAll"])

    def salvar_dados_csv_yard(self, episode, agent_id, yard_value):
        # Adiciona uma nova linha ao arquivo CSV
        with open(self.csv_path_yard, mode='a', newline='') as arquivo_csv:
            escritor_csv = csv.writer(arquivo_csv)
            escritor_csv.writerow([episode, agent_id, yard_value])
    
    def salvar_dados_csv_action(self, episode, agent_id, yard_value):
        # Adiciona uma nova linha ao arquivo CSV
        with open(self.csv_path_action, mode='a', newline='') as arquivo_csv:
            escritor_csv = csv.writer(arquivo_csv)
            escritor_csv.writerow([episode, agent_id, yard_value])
    
    def salvar_dados_csv_reject(self, episode, agent_id, yard_value):
        # Adiciona uma nova linha ao arquivo CSV
        with open(self.csv_path_reject, mode='a', newline='') as arquivo_csv:
            escritor_csv = csv.writer(arquivo_csv)
            escritor_csv.writerow([episode, agent_id, yard_value])
    
    def salvar_dados_csv_reject_w_wast(self, episode, agent_id, yard_value):
        # Adiciona uma nova linha ao arquivo CSV
        with open(self.csv_path_rejectwwast, mode='a', newline='') as arquivo_csv:
            escritor_csv = csv.writer(arquivo_csv)
            escritor_csv.writerow([episode, agent_id, yard_value])
    
    def salvar_dados_csv_reject_all(self, episode, yard_value):
        # Adiciona uma nova linha ao arquivo CSV
        with open(self.csv_path_rejectall, mode='a', newline='') as arquivo_csv:
            escritor_csv = csv.writer(arquivo_csv)
            escritor_csv.writerow([episode, yard_value])

    def salvar_dados_csv_acept(self, episode,agent_id, yard_value):
        # Adiciona uma nova linha ao arquivo CSV
        with open(self.csv_path_rejectall, mode='a', newline='') as arquivo_csv:
            escritor_csv = csv.writer(arquivo_csv)
            escritor_csv.writerow([episode,agent_id, yard_value])
        
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

    def on_episode_step(self, *, worker: RolloutWorker, base_env, policies=None, episode, env_index=None, **kwargs):
        self.step+=1
        for agent_id in episode.get_agents():
            agent_info = episode.last_info_for(agent_id)
            
            yard_value = agent_info.get("yard")
            action_value = agent_info.get("acoes")
            reject_value = agent_info.get("reject")
            rejectw_value = agent_info.get("reject_w_west")
            acept_value = agent_info.get("acept_reject")

            self.reject[agent_id] = reject_value
            self.reject_w[agent_id] = rejectw_value

            self.yard_hist[agent_id] = yard_value
            self.action_hist[agent_id] = action_value

            self.acept[agent_id] = acept_value

            self.acept_produced[agent_id] = agent_info.get("produced_reject")
            self.acept_yard[agent_id] = agent_info.get("yard_reject")

            episode.custom_metrics[f"Yard agent {agent_id}"] = yard_value
            episode.custom_metrics[f"Action agent {agent_id}"] = action_value
            episode.custom_metrics[f"Acept_reject agent {agent_id}"] = acept_value

            # Verifica se o valor de "yard" está presente
            if yard_value is not None:
                # Salva os dados em CSV
                self.salvar_dados_csv_yard(self.step, agent_id, yard_value)
            
            if action_value is not None:
                self.salvar_dados_csv_action(self.step, agent_id, action_value)
        
            self.reject_all = agent_info.get("reject_all")
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env, policies, episode, env_index, **kwargs) -> None:
        episode.custom_metrics[f"Reject_All"] = self.reject_all
        for agent_id in self.yard_hist:
            episode.custom_metrics[f"reject agent {agent_id}"] = self.reject[agent_id]
            episode.custom_metrics[f"Reject_w_wast {agent_id}"] = self.reject_w[agent_id]
            episode.custom_metrics[f"Acept_produced_agent {agent_id}"] = self.acept_produced[agent_id]
            episode.custom_metrics[f"Acept_Yard_agent {agent_id}"] = self.acept_yard[agent_id]

            self.numepisode += 1
            self.sum_acept += self.acept[agent_id]
            self.sum_reject += self.reject[agent_id]

            self.salvar_dados_csv_reject(self.numepisode, agent_id, self.reject[agent_id])
            self.salvar_dados_csv_reject_w_wast(self.numepisode, agent_id, self.reject_w[agent_id])
            self.salvar_dados_csv_acept(self.numepisode, agent_id, self.acept[agent_id])

        self.salvar_dados_csv_reject_all(self.numepisode, self.reject_all)
        episode.custom_metrics[f"Sum_Acept"] = self.sum_acept
        episode.custom_metrics[f"Sum_Reject"] = self.sum_reject