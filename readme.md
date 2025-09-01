# EdgeSim-RL

A reinforcement learning framework for edge computing resource management built on top of EdgeSimPy and Ray RLlib.

## Overview

EdgeSim-RL provides a modular and extensible framework for training reinforcement learning agents to optimize resource allocation and service migration in edge computing environments. Unlike direct RL implementations, this framework leverages Ray RLlib for scalable and efficient training.

## Features

- **Modular Architecture**: Clean separation of environments, rewards, and algorithms
- **RLlib Integration**: Leverages Ray's powerful distributed RL library
- **Extensible Design**: Easy to add new algorithms, reward functions, and environment configurations
- **EdgeSimPy Compatible**: Seamlessly works with EdgeSimPy simulation framework

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/edgesim-rl.git
cd edgesim-rl

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### Basic Training

```bash
# Train a DQN agent with default configuration
python src/train.py --config configs/dqn_config.yaml

# Train a PPO agent with default configuration
python src/train.py --config configs/ppo_config.yaml

# Train with custom parameters
python src/train.py --config configs/dqn_config.yaml --num-workers 4 --num-gpus 1
```

### Custom Configuration

Edit `configs/dqn_config.yaml` to modify:
- Environment parameters
- Algorithm hyperparameters  
- Training settings
- Reward function parameters

## Architecture

### Environment (`src/envs/`)
- `base_env.py`: Abstract base environment class
- `edge_env.py`: EdgeSimPy environment wrapper implementing Gym interface

### Rewards (`src/rewards/`)
- `power_reward.py`: Power consumption-based reward functions

### Algorithms (`src/algorithms/`)
- `trainer.py`: RLlib trainer wrapper with DQN and PPO configurations

### Utils (`src/utils/`)
- `metrics.py`: Metrics collection and logging utilities

## Extending the Framework

### Adding New Algorithms

Create a new configuration in `configs/` and update the trainer:

```python
# In src/algorithms/trainer.py
def get_algorithm_config(algorithm_name: str, config: dict):
    if algorithm_name == "DQN":
        return get_dqn_config(config)
    elif algorithm_name == "YOUR_ALGORITHM":
        return get_your_algorithm_config(config)
```

### Adding New Reward Functions

Create a new reward class in `src/rewards/`:

```python
# src/rewards/custom_reward.py
class CustomReward:
    def calculate(self, state, action, next_state, info):
        # Your reward logic
        return reward
```

### Modifying the Environment

Extend the `BaseEdgeEnv` class to create custom environments with different state/action spaces.

## Dataset Format

The framework uses EdgeSimPy's JSON dataset format. See `datasets/sample_dataset.json` for an example structure.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{edgesim-rl,
  title = {EdgeSim-RL: Reinforcement Learning Framework for Edge Computing},
  year = {2024},
  url = {https://github.com/yourusername/edgesim-rl}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.