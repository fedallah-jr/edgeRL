"""EdgeSimPy environment wrapper for reinforcement learning."""

import numpy as np
import gym
from gym import spaces
from typing import Dict, Any, Tuple, List, Optional
from edge_sim_py import *
from .base_env import BaseEdgeEnv
from ..rewards.power_reward import PowerReward


class EdgeEnv(BaseEdgeEnv):
    """Gym environment wrapper for EdgeSimPy simulations.
    
    This environment handles service migration decisions in edge computing
    infrastructures, optimizing for power consumption and QoS.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize EdgeEnv.
        
        Args:
            config: Environment configuration
        """
        super().__init__(config)
        
        # Initialize simulator
        self.simulator = None
        self.dataset_path = config.get("dataset_path", "datasets/sample_dataset.json")
        self.tick_duration = config.get("tick_duration", 1)
        self.tick_unit = config.get("tick_unit", "seconds")
        
        # State configuration
        self.state_config = config.get("state", {})
        self.include_server_metrics = self.state_config.get("include_server_metrics", True)
        self.include_service_metrics = self.state_config.get("include_service_metrics", True)
        self.normalize_state = self.state_config.get("normalize", True)
        
        # Action configuration
        self.action_config = config.get("action", {})
        self.allow_no_migration = self.action_config.get("allow_no_migration", True)
        
        # Reward configuration
        reward_config = config.get("reward", {})
        reward_type = reward_config.get("type", "power")
        
        if reward_type == "power":
            self.reward_calculator = PowerReward(reward_config)
        else:
            # Default to power reward
            self.reward_calculator = PowerReward(reward_config)
        
        # Initialize environment
        self._init_simulator()
        self._init_spaces()
        
        # Migration tracking
        self.current_service_idx = 0
        self.services_to_migrate = []
        self.last_state = None
        
    def _init_simulator(self):
        """Initialize EdgeSimPy simulator."""
        # Create simulator with our custom algorithm
        self.simulator = Simulator(
            tick_duration=self.tick_duration,
            tick_unit=self.tick_unit,
            stopping_criterion=lambda model: model.schedule.steps >= self.max_steps,
            resource_management_algorithm=self._rl_algorithm,
            dump_interval=float('inf')  # Disable automatic dumping
        )
        
        # Load dataset
        self.simulator.initialize(input_file=self.dataset_path)
        
    def _init_spaces(self):
        """Initialize observation and action spaces."""
        # Get environment dimensions
        self.num_servers = EdgeServer.count()
        self.num_services = Service.count()
        
        # Define observation space
        obs_dim = self._get_observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Define action space (discrete: which server to migrate to)
        # Add 1 for no-migration action if enabled
        action_dim = self.num_servers
        if self.allow_no_migration:
            action_dim += 1
            
        self.action_space = spaces.Discrete(action_dim)
        
    def _get_observation_dim(self) -> int:
        """Calculate observation space dimensionality."""
        dim = 0
        
        if self.include_server_metrics:
            # CPU, memory, disk, power for each server
            dim += self.num_servers * 4
            
        if self.include_service_metrics:
            # Current server one-hot encoding + resource demands
            dim += self.num_servers + 3  # one-hot + cpu + mem + state_size
            
        return dim
        
    def _rl_algorithm(self, parameters: Dict):
        """Custom resource management algorithm for RL.
        
        This method is called by EdgeSimPy at each simulation step.
        """
        # Collect services that need migration
        self.services_to_migrate = [
            service for service in Service.all()
            if not service.being_provisioned and service.server is None
        ]
        
        # Reset service index for new round
        self.current_service_idx = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        super().reset()
        
        # Reset simulator
        self._init_simulator()
        
        # Reset tracking variables
        self.current_service_idx = 0
        self.services_to_migrate = []
        self.last_state = None
        
        # Get initial observation
        return self.get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step.
        
        Args:
            action: Server index to migrate current service to
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        info = {}
        
        # Get current state before action
        state_before = self.get_state()
        
        # Check if we have services to process
        if self.current_service_idx < len(self.services_to_migrate):
            service = self.services_to_migrate[self.current_service_idx]
            
            # Execute action (migration)
            if self.is_valid_action(action):
                if action < self.num_servers:
                    # Migrate to specified server
                    target_server = EdgeServer.all()[action]
                    if target_server.has_capacity_to_host(service):
                        service.provision(target_server=target_server)
                        info['migration'] = True
                        info['valid_action'] = True
                    else:
                        info['valid_action'] = False
                else:
                    # No migration action
                    info['migration'] = False
                    info['valid_action'] = True
            else:
                info['valid_action'] = False
            
            self.current_service_idx += 1
        
        # If we've processed all services, advance simulation
        if self.current_service_idx >= len(self.services_to_migrate):
            self.simulator.step()
            self.current_step += 1
        
        # Get new state
        state_after = self.get_state()
        
        # Calculate reward
        reward = self.reward_calculator.calculate(
            state_before, action, state_after, info
        )
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Store current state
        self.last_state = state_after
        
        return state_after, reward, done, info
    
    def get_state(self) -> np.ndarray:
        """Get current environment state.
        
        Returns:
            State observation as numpy array
        """
        state = []
        
        if self.include_server_metrics:
            # Add server metrics
            for server in EdgeServer.all():
                cpu_util = server.cpu_demand / server.cpu if server.cpu > 0 else 0
                mem_util = server.memory_demand / server.memory if server.memory > 0 else 0
                disk_util = server.disk_demand / server.disk if server.disk > 0 else 0
                power = server.get_power_consumption()
                
                if self.normalize_state:
                    # Normalize utilization is already 0-1
                    # Normalize power (assuming max power from parameters)
                    if hasattr(server, 'power_model_parameters'):
                        max_power = server.power_model_parameters.get('max_power_consumption', 100)
                        power = power / max_power if max_power > 0 else 0
                
                state.extend([cpu_util, mem_util, disk_util, power])
        
        if self.include_service_metrics and self.current_service_idx < len(self.services_to_migrate):
            service = self.services_to_migrate[self.current_service_idx]
            
            # One-hot encoding for current server
            server_one_hot = [0] * self.num_servers
            if service.server:
                server_idx = EdgeServer.all().index(service.server)
                server_one_hot[server_idx] = 1
            state.extend(server_one_hot)
            
            # Service resource demands (normalized if configured)
            if self.normalize_state:
                max_cpu = max(s.cpu for s in EdgeServer.all())
                max_mem = max(s.memory for s in EdgeServer.all())
                cpu_demand = service.cpu_demand / max_cpu if max_cpu > 0 else 0
                mem_demand = service.memory_demand / max_mem if max_mem > 0 else 0
                state_size = service.state / 1000.0  # Normalize to MB scale
            else:
                cpu_demand = service.cpu_demand
                mem_demand = service.memory_demand
                state_size = service.state
                
            state.extend([cpu_demand, mem_demand, state_size])
        elif self.include_service_metrics:
            # Pad with zeros if no service to migrate
            state.extend([0] * (self.num_servers + 3))
        
        return np.array(state, dtype=np.float32)
    
    def is_valid_action(self, action: int) -> bool:
        """Check if action is valid in current state.
        
        Args:
            action: Server index or no-migration action
            
        Returns:
            True if action is valid
        """
        if action >= self.action_space.n:
            return False
        
        # No-migration is always valid if enabled
        if self.allow_no_migration and action == self.num_servers:
            return True
        
        # Check if migration to server is valid
        if action < self.num_servers and self.current_service_idx < len(self.services_to_migrate):
            service = self.services_to_migrate[self.current_service_idx]
            target_server = EdgeServer.all()[action]
            return target_server.has_capacity_to_host(service)
        
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get additional environment information.
        
        Returns:
            Dictionary with environment metrics
        """
        total_power = sum(server.get_power_consumption() for server in EdgeServer.all())
        
        info = {
            'total_power': total_power,
            'num_migrations': len([s for s in Service.all() if s.being_provisioned]),
            'current_step': self.current_step,
            'services_to_migrate': len(self.services_to_migrate)
        }
        
        return info