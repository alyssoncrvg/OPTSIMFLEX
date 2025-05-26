# OPTSIMFLEX: Multi-Agent Simulation of Manufacturing Networks for Industry 4.0

## Project Overview

OPTSIMFLEX (Optimized Simulation for Flexible Manufacturing Systems) is a Python-based simulation environment designed to model and optimize industrial production in the context of Industry 4.0. This project, as described in the accompanying research paper "Multi-Agent Simulation of Manufacturing Networks: Autonomous and Adaptive Decision-Making for Industry 4.0", integrates intelligent agents and reinforcement learning to enhance efficiency and flexibility in decision-making within autonomous manufacturing units.

The core of the project is a multi-agent system where each agent represents a manufacturing unit. These agents make autonomous decisions regarding demand acceptance and processing, aiming to balance multiple criteria:
* **Profit**: Maximizing economic returns or minimizing costs.
* **Sustainability**: Reducing environmental impact, such as minimizing energy consumption and waste from inefficient production or large storage yards.
* **Variability (Customization)**: Adapting products or services to meet specific customer preferences.

The system allows for real-time adaptation and facilitates communication between units, enabling the redistribution of tasks to minimize rejections and boost overall productivity.

## System Architecture

The simulation environment models a network of self-organizing manufacturing units (SOMN). Each unit is controlled by a reinforcement learning agent that learns local acceptance policies. Demands (product requests) arrive and are routed through the network based on their characteristics and the current state of the manufacturing units.

Key architectural features include:
* **Multi-Agent System**: The environment supports multiple autonomous agents, each managing a manufacturing unit.
* **Multi-Criteria Decision Making**: Agents' decisions are guided by a framework balancing profit, sustainability, and variability.
* **Demand Prioritization and Routing**: Arriving demands are classified (e.g., Profit-Driven, Eco-Certified, Custom-Made) and placed into specialized queues for processing by the appropriate unit. Agents dynamically manage this routing logic.
* **Inter-Agent Communication & Demand Transfer**: If a unit rejects a demand, the system attempts to transfer it to another suitable unit within the network. This minimizes overall rejections.
* **Dynamic Adaptation**: The system adapts to uncertainties like raw material availability, plant workload, and delays.
* **Fallback System**: A three-tier fallback system with cascading fail-safes is implemented to handle demand processing.

## Core Components (Python Files)

The project is structured into several Python modules within the `Ambiente_SOMN` directory:

* **`Somn.py`**: This is the main environment class. It implements the `PettingZoo` Parallel API for multi-agent reinforcement learning. It manages the agents, the overall simulation state, demand processing lifecycle, reward calculations, and interactions between different components.
* **`Demand.py`**: Defines the `Demand` class, representing customer orders or product requests. Each demand has various attributes like amount (`AM`), price (`PR`), cost (`CO`), lead time (`LT`), sustainability (`SU`), variability (`VA`), features/resource matrix (`FT`), and status (`ST`).
* **`Yard.py`**: Implements the `Yard` class, which models the storage space for each manufacturing unit. It keeps track of stored products and their characteristics.
* **`Statistcs.py`**: A simple class to track statistics like the number of demands, load, rejections, and production with waste.
* **`MyCallbacks.py`**: Implements custom callbacks for Ray RLlib, used for logging detailed information during training and evaluation, such as yard status, agent actions, rejections, and custom metrics to CSV files and Weights & Biases.
* **`fuzzer_control.py`**: Contains the `controle` function, which implements a fuzzy logic controller. This controller helps in balancing different objectives by adjusting decision parameters based on error and derivative of error for different criteria (e.g., variability vs. profit, sustainability vs. profit).
* **`make_env.py`**: A utility script to create an instance of the `Somn` environment with specified parameters.
* **`saveDatas.py`**: Contains functions to save data related to priority queues and chosen queues during simulation to CSV files.
* **`Poisson.py`**: Used for generating random numbers from a Poisson distribution, likely to model uncertainties such as delays or real lead times.

Other important files include:
* **`train.py`**: The main script for training the reinforcement learning agents using Ray RLlib and PPO.
* **`JobShop/JobShop.py`**: Contains a `JobShop` class using `ortools.sat.python` for solving job shop scheduling problems.

## Key Concepts and Logic

### Demand Lifecycle & Statuses

A demand progresses through several states within the simulation (based on `Somn.py` and `Demand.py`):
* **`REJECTED_W_WASTE` (-2)**: Demand produced but couldn't be stored (e.g., yard full) and resulted in waste.
* **`FREE` (-1)**: The demand slot is available for a new demand. (In `Somn.py`, the paper refers to "transferred" as -1 for demand status).
* **`RECEIVED` (0)**: Demand has arrived at a unit.
* **`READY` (1)**: Demand has the necessary raw materials allocated and is ready for production planning.
* **`REJECTED` (2)**: Demand is rejected by the current unit (may be transferred).
* **`PRODUCTION` (3)**: Demand is currently being produced.
* **`STORED` (4)**: Demand was produced but missed its delivery deadline and is now in the yard.
* **`DELIVERED` (5)**: Demand was successfully produced and delivered on time.

### Environment Variables and State Representation

The `Somn` environment keeps track of several variables crucial for decision-making:
* **Stock Balance (`BA`)**: For M raw materials at each unit.
* **Incoming Materials (`IN`)**: Ordered but not yet received.
* **Outgoing/Allocated Materials (`OU`)**: Allocated to production.
* **Availability (`AV`)**: Calculated as `BA + IN - OU`.
* **Demand Attributes**: As defined in `Demand.py`, including:
    * `AM`: Amount.
    * `FT`: Feature/Resource matrix (vector of raw materials/machine times).
    * `LT`: Expected lead time, calculated by `fun_tau(AM_d * FT_d)`.
    * `VA`: Variability, calculated by `fun_upsilon(FT_d)`.
    * `SU`: Sustainability, calculated by `fun_sigma(FT_d^-1)`. (Note: `Demand.py` uses `1 - fun_sigma()`)
    * `CO`: Cost, calculated based on `AM`, `FT`, and `EU` (cost per unit of feature).
    * `PR`: Net profit per unit.
    * `DI`, `DO`: Date of input and delivery deadline.
    * `TP`: Realized production time.
    * `EU`: Cost per unit of each feature/raw material (defined in `Somn.py` and `Demand.py`).
* **Machine/Unit Attributes**: `MT` (material transformation times/capabilities), `EU` (energy/cost unit for features) (defined in `Somn.py`).
* **Yard State**: Current occupancy and capacity (managed by `Yard.py`).

The **observation space** for each agent is a dictionary containing normalized values for:
* `time`: Current simulation time.
* `MT`, `EU`, `BA`, `IN`, `OU`: Unit-specific material and cost vectors.
* `DE_state`: An array representing the state of all N demands associated with the agent (normalized attributes like ST, DI, DO, TP, PR, CO, AM, SP, PE, VA, SU, F, LT, real_LT, atraso_real, action, err).
* `FT_state`: An array representing the feature vectors (`FT`) of all N demands.
* `yard`: Normalized yard occupancy.
* `load`: Normalized current production load of the unit.
(Details derived from `Somn.py`'s `observation_spaces` and `step` method's observation construction).

### Action Space

The **action space** for each agent is discrete, representing the "allowed lateness" or adjustment to the lead time an agent decides for a demand in the `READY` state. `action_spaces = {f"{i}" : spaces.Discrete(self.MAX_ATRASO) for i in range(numAgents)}` (from `Somn.py`). This action influences whether a demand will be produced or rejected based on its deadline.

### Reward System

The reward function is designed to align agent behavior with the multi-criteria objectives.
* **Positive Rewards**: Granted when a demand is successfully `DELIVERED` or `STORED`. The specific reward value depends on the unit's priority (profit, variability, or sustainability) as defined in Equation 11 of the paper:
    * If Profit priority: $RW_{pr} = AM \cdot PR$
    * If Variability priority: $RW_{va} = AM \cdot PR \cdot VA$
    * If Sustainability priority: $RW_{su} = AM \cdot PR \cdot SU$
    (Implemented in `Somn.py` in the `step` method based on the `fila` (queue) chosen, which implies the current objective).
* **Penalties (`PE_d`)**: Applied for negative outcomes:
    * **Storing a delayed demand**: Penalized based on yard occupancy and an environmental tax (`ET`). (In `Somn.py`, `store()`: `self.totPenalty += (self.YA[agent].cont/self.YA[agent].space) * self.DE[agent][i].AM * self.DE[agent][i].CO * self.tx_penalidade; self.totPenalty += self.DE[agent][i].AM * self.DE[agent][i].PR * tx_ambiente * 0.01`)
    * **Demand resulting in waste (`REJECTED_W_WASTE`)**: Incurs a penalty related to the demand's cost and an environmental tax. (In `Somn.py`, `reject_w_waste()`: `self.totPenalty += self.DE[agent][i].AM * self.DE[agent][i].CO * self.tx_penalidade; self.totPenalty += self.DE[agent][i].AM * self.DE[agent][i].PR * tx_ambiente * 0.1`)
    * **Rejecting a demand (`REJECTED`)**: May incur a smaller penalty. (In `Somn.py`, `reject()`: `self.totPenalty += self.DE[agent][i].AM * self.DE[agent][i].PR * tx_ambiente * 0.005`)

The overall reward for an agent in a step is `self.totReward - self.totPenalty` (derived from `Somn.py`).

### Decision Rules and Demand Handling

1.  **Order Reception and Matching (`order_receive_and_match` in `Somn.py`)**:
    * New raw materials (`MT`) are received.
    * New demands are read if slots are `FREE` (`ST == -1`).
    * The system checks if an incoming demand (`ST == 0`) can be fulfilled by a product already in the `Yard` (matching `mask_FT`). If so, demand is `DELIVERED`.
    * Stock balance (`BA`) is updated.
    * The system checks if current `BA` covers features (`FT`) of `RECEIVED` demands (`stock_covers_demand` in `Somn.py`).
        * If covered, demand status becomes `READY` (1), and it's added to priority queues (`Somn.priorq`).
        * If not covered, necessary raw materials are ordered (added to `IN`).

2.  **Priority Queues and Agent Focus (`balance`, `plan` in `Somn.py`)**:
    * Each agent maintains multiple priority queues (`Somn.priorq`), one for each objective (Profit, Variability, Sustainability). Demands in the `READY` state are added to these queues.
    * The priority indices from the paper (Equations 5, 6, 7) are used to sort demands within these queues:
        * $h_{1,d} = \frac{1}{AM_{d} \cdot PR_d}$ (Profit, implemented as `1/(self.DE[agent][i].AM * self.DE[agent][i].PR)`)
        * $h_{2,d} = 1 - VA_d$ (Variability)
        * $h_{3,d} = 1 - SU_d$ (Sustainability)
    * The `plan` method takes the highest priority demand from a selected queue (`fila`). The selection of `fila` depends on the operational mode (see sections below on Fuzzy Controllers or Fixed Objectives).
    * The agent's `action` (allowed lateness) is applied. If `Demand.DO > (time + Demand.LT + action)`, the demand is set to `PRODUCTION` (3). Otherwise, it's `REJECTED` (2).

3.  **Production, Dispatch, Storage, Rejection (`produce`, `dispatch`, `store`, `reject`, `reject_w_waste` in `Somn.py`)**:
    * `produce`: If a demand is in `PRODUCTION` and its `TP` (realized production time) is less than current time `t`:
        * If `t < Demand.DO`, it's `DELIVERED` (5).
        * Else, it's `STORED` (4). If the yard is full, it becomes `REJECTED_W_WASTE` (-2).
    * `dispatch`: If `DELIVERED`, calculate reward, set demand slot to `FREE`.
    * `store`: If `STORED`, calculate penalty, set demand slot to `FREE`.
    * `reject`: If `REJECTED`, calculate penalty, set demand slot to `FREE`. The rejected demand is added to `self.rejecteds` for potential transfer.
    * `reject_w_waste`: If `REJECTED_W_WASTE`, calculate penalty, set demand slot to `FREE`.

4.  **Demand Transfer (`rejected` and `destine` methods in `Somn.py`)**:
    * The paper describes a `destine` function (Equation 8) to find a new agent for a rejected demand based on matching the demand's strongest attribute ($h_{1,d}, h_{2,d}, h_{3,d}$) with an agent's priority/objective and choosing the agent with the shortest queue for that objective.
    * The `Somn.py`'s `rejected` method processes a rejected demand and calls `self.destine` to attempt re-allocation.
    * `self.destine` uses the agent's predefined objective (`self.objetivo`) to find a suitable new agent.
    * A demand keeps track of agents that have rejected it (`demand.rejects`).
    * If a demand is rejected by all agents, it's counted in `self.demands_rejects_all`.

## Operational Modes and Experimental Features

### Special Note: Fuzzy Controllers (Dynamic Objectives)

This codebase includes experimental support for **Fuzzy Logic Controllers** to dynamically balance agent objectives:

* The `Ambiente_SOMN/fuzzer_control.py` script implements a fuzzy logic controller (`controle` function).
* This controller can be used in `Somn.py`'s `balance` method to dynamically choose which priority queue (`fila` representing profit, variability, or sustainability) an agent should focus on, based on global system performance.
* **Current Status in `Somn.py`**: In the `step` method of `Somn.py` (around line 1028), the call to the fuzzy logic-based balancing is currently **commented out**:
    ```python
    # fila = self.balance(agent, self.safe_mean(self.lucro), self.safe_mean(self.sustentabilidade), self.safe_mean(self.variabilidade))
    fila = random.randint(0, 2) # Currently using random selection for 'fila'
    ```
    To enable the fuzzy controller for dynamic queue selection, uncomment the `self.balance(...)` line and comment out or remove the `fila = random.randint(0, 2)` line.

### Executing with Fixed Agent Objectives (Legacy Demand Sharing - As per Article)

The research paper "Multi-Agent Simulation of Manufacturing Networks: Autonomous and Adaptive Decision-Making for Industry 4.0" also describes a demand sharing mechanism based on agents having predefined, **fixed strategic objectives**.

To configure and run the simulation in this mode:

1.  **Ensure Fixed Objectives are Set:**
    * In `Ambiente_SOMN/Somn.py`, the `__init__` method (around line 181) sets `self.objetivo` cyclically:
        ```python
        # linha aprox. 181 em Somn.py
        self.objetivo = [i % 3 for i in range(num_agents)] #0-lucro, 1-variabilidade, 2-sustentabilidade
        ```
        This assigns objectives (0 for Profit, 1 for Variability, 2 for Sustainability) to agents. You can modify this line if a different static assignment is needed (e.g., `self.objetivo = [0,0,1,1,2,2]` for 6 agents).

2.  **Use Fixed Objective for Queue Selection:**
    * In `Somn.py`'s `step` method (around line 1028-1032), modify the `fila` selection to use the agent's fixed objective:
        ```python
        # Comment out or remove these lines:
        # fila = self.balance(agent, self.safe_mean(self.lucro), self.safe_mean(self.sustentabilidade), self.safe_mean(self.variabilidade))
        # fila = random.randint(0, 2)

        # Add this line to use the fixed objective:
        agent_id_numeric = self.agents_id[agent] # Get the numeric ID of the agent
        fila = self.objetivo[agent_id_numeric]
        ```

3.  **Activate Demand Sharing Mechanism:**
    * The demand sharing mechanism relies on processing rejected demands. In `Somn.py`'s `step` method (around line 1155), ensure the loop for processing `self.rejecteds` is active (uncommented):
        ```python
        # Ensure this loop is active for demand sharing:
        self.aux = list(self.rejecteds) # Create a copy to iterate over if self.rejected might modify self.rejecteds
        self.rejecteds.clear() # Clear the original list early or after processing
        for i_demanda_rejeitada in self.aux:
            self.rejected(i_demanda_rejeitada) # This calls self.destine() which uses self.objetivo
        ```
        The `self.destine()` method (called by `self.rejected()`) uses `self.objetivo[i]` (where `i` is an agent index) to find a new destination for the rejected demand, aligning with the paper's logic.

By combining fixed objectives with an active demand sharing loop, the simulation will operate as described in the article regarding demand transfers based on specialized agent roles.

## Setup and Installation

Follow these steps to set up the environment and run the simulation (based on `Readme.md`):

1.  **Create and Activate a Virtual Environment**:
    ```bash
    python3 -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

2.  **Install Dependencies**:
    Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Login to Weights & Biases (Wandb)**:
    This project uses Wandb for experiment tracking and logging. You'll need an account and an API key.
    ```bash
    wandb login YOUR_API_KEY
    ```
    Replace `YOUR_API_KEY` with your actual key. The `Instalando WandB/login.py` script can also be used if you prefer to run a Python script for login.

## How to Run

### Training Agents

* The primary script for training is `train.py`.
* It uses Ray RLlib's PPO (Proximal Policy Optimization) algorithm.
* The environment `SOMN` is registered with Ray.
* `MyCallbacks` is used for custom logging during training.
* The script sets up a PPOConfig, specifying the environment, callbacks, resources (GPUs), and multi-agent configuration.
* The `policy_mapping_fn` in `train.py` currently maps each agent to its own ID, meaning each agent learns its own policy.
* Training is run using `tune.Tuner`.
* Results, including metrics and model checkpoints, are logged to Wandb and locally.

To start training:
```bash
python train.py