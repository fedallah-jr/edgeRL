"""EdgeSimPy environment wrapper for reinforcement learning."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, List, Optional
from edge_sim_py import *
from edge_sim_py.activation_schedulers.random_scheduler import RandomScheduler
from .base_env import BaseEdgeEnv
from ..rewards.power_reward import PowerReward
from ..rewards.edge_aisim_reward import EdgeAISIMReward


class EdgeEnv(BaseEdgeEnv):
    """Gym environment wrapper for EdgeSimPy simulations.
    
    This environment handles service migration decisions in edge computing
    infrastructures, optimizing for power consumption and QoS.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize EdgeEnv.
        
        Args:
            config: Environment configuration
        """
        # Handle empty config
        if config is None:
            config = {}
            
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
        
        # Randomization and scheduler configuration
        self.randomize_initial_placement = bool(config.get("randomize_initial_placement", True))
        self.use_random_scheduler = bool(config.get("use_random_scheduler", True))
        
        # Reward configuration
        reward_config = config.get("reward", {})
        reward_type = reward_config.get("type", "power")
        
        if reward_type == "power":
            self.reward_calculator = PowerReward(reward_config)
        elif reward_type in ("edgeaisim", "inverse_power", "edge_ai_sim"):
            # EdgeAISIM-style immediate reward (sum of inverse power across servers)
            self.reward_calculator = EdgeAISIMReward(reward_config)
        else:
            # Default to EdgeAISIM reward when unknown, then fall back to power reward
            try:
                self.reward_calculator = EdgeAISIMReward(reward_config)
            except Exception:
                self.reward_calculator = PowerReward(reward_config)
        
        # Initialize environment
        self._init_simulator()
        self._init_spaces()
        
        # Migration tracking
        self.current_service_idx = 0
        self.services_to_migrate = []
        self.last_state = None
        self.migration_scheduled = False
        
    def _init_simulator(self):
        """Initialize EdgeSimPy simulator."""
        try:
            # Create simulator with our custom algorithm
            sim_kwargs = {
                "tick_duration": self.tick_duration,
                "tick_unit": self.tick_unit,
                "stopping_criterion": lambda model: model.schedule.steps >= self.max_steps,
                "resource_management_algorithm": self._rl_algorithm,
                "dump_interval": float('inf'),  # Disable automatic dumping
            }
            if self.use_random_scheduler:
                # Use EdgeSimPy's RandomScheduler to introduce stochastic agent activation order
                sim_kwargs["scheduler"] = RandomScheduler
            self.simulator = Simulator(**sim_kwargs)
            
            # Load dataset
            self.simulator.initialize(input_file=self.dataset_path)
            
            # Initialize services on servers to avoid initial empty state
            # This is a workaround for the initial service placement
            self._initial_service_placement()
            
        except Exception as e:
            print(f"Error initializing simulator: {e}")
            raise
            
    def _initial_service_placement(self):
        """Perform initial placement of services on servers.
        
        If randomize_initial_placement is True (default), services will be placed in a randomized order and
        candidate servers will be checked in a randomized order (capacity-respecting). Otherwise, a simple
        deterministic round-robin is used.
        """
        try:
            servers = EdgeServer.all()
            services = Service.all()
            
            if not servers or not services:
                return
            
            if self.randomize_initial_placement:
                # Randomize service order and server order for placement attempts
                try:
                    service_indices = np.arange(len(services))
                    np.random.shuffle(service_indices)
                    server_indices = np.arange(len(servers))
                    np.random.shuffle(server_indices)
                except Exception:
                    # Fallback in case numpy RNG fails for any reason
                    service_indices = list(range(len(services)))
                    server_indices = list(range(len(servers)))
                
                for si in service_indices:
                    service = services[si]
                    if service.server is None:
                        placed = False
                        for sj in server_indices:
                            server = servers[int(sj)]
                            if server.has_capacity_to_host(service):
                                try:
                                    service.provision(target_server=server)
                                    placed = True
                                    break
                                except Exception:
                                    continue
                        # Optional: rotate server order a bit to avoid bias
                        try:
                            if len(server_indices) > 1:
                                # move first index to the end
                                first = server_indices[0]
                                server_indices = np.append(server_indices[1:], first)
                        except Exception:
                            pass
            else:
                # Simple round-robin initial placement (deterministic)
                server_idx = 0
                for service in services:
                    if service.server is None:
                        # Try to find a server with capacity
                        placed = False
                        attempts = 0
                        while not placed and attempts < len(servers):
                            server = servers[server_idx % len(servers)]
                            if server.has_capacity_to_host(service):
                                try:
                                    service.provision(target_server=server)
                                    placed = True
                                except Exception:
                                    pass
                            server_idx += 1
                            attempts += 1
                        
        except Exception as e:
            # Silent fail - initial placement is optional
            pass
        
    def _init_spaces(self):
        """Initialize observation and action spaces."""
        try:
            # Get environment dimensions
            self.num_servers = EdgeServer.count()
            self.num_services = Service.count()
            
            # Handle edge case where no servers or services exist
            if self.num_servers == 0:
                self.num_servers = 1
            if self.num_services == 0:
                self.num_services = 1
            
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
            
        except Exception as e:
            print(f"Error initializing spaces: {e}")
            # Set default spaces
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10,),
                dtype=np.float32
            )
            self.action_space = spaces.Discrete(2)
        
    def _get_observation_dim(self) -> int:
        """Calculate observation space dimensionality."""
        dim = 0
        
        if self.include_server_metrics:
            # CPU, memory, disk, power for each server
            dim += self.num_servers * 4
            
        if self.include_service_metrics:
            # Current server one-hot encoding + resource demands
            dim += self.num_servers + 3  # one-hot + cpu + mem + state_size
            
        # Ensure minimum dimension
        return max(dim, 1)
        
    def _rl_algorithm(self, parameters: Dict):
        """Custom resource management algorithm for RL.
        
        This method is called by EdgeSimPy at each simulation step.
        Align with EdgeAISIM by considering all services that are not being
        provisioned as candidates for migration decisions.
        """
        try:
            # Collect services that are eligible for migration decisions
            self.services_to_migrate = [
                service for service in Service.all()
                if not service.being_provisioned
            ]
            
            # Set flag to indicate migration is needed
            if self.services_to_migrate:
                self.migration_scheduled = True
            
            # Reset service index for new round
            self.current_service_idx = 0
            
        except Exception as e:
            # Handle any errors gracefully
            self.services_to_migrate = []
            self.current_service_idx = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        try:
            # Reset simulator
            self._init_simulator()
            
            # Reset tracking variables
            self.current_service_idx = 0
            self.services_to_migrate = []
            self.last_state = None
            self.migration_scheduled = False
            
            # Reset reward calculator
            if hasattr(self.reward_calculator, 'reset'):
                self.reward_calculator.reset()
            
            # Get initial observation and info
            obs = self.get_state()
            info = self.get_info()
            
            return obs, info
            
        except Exception as e:
            print(f"Error in reset: {e}")
            # Return default observation
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step.
        
        Args:
            action: Server index to migrate current service to
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        info = {}
        
        try:
            # Ensure we have a fresh list of services to process for this scheduling round
            if not self.services_to_migrate:
                try:
                    self.services_to_migrate = [
                        s for s in Service.all() if not s.being_provisioned
                    ]
                    self.current_service_idx = 0
                except:
                    self.services_to_migrate = []
                    self.current_service_idx = 0

            # Get current state before action
            state_before = self.get_state()
            
            # Check if we have services to process
            decided_service_id = None
            pattern_choice = -1  # -1 means no migration for pattern accounting
            if self.current_service_idx < len(self.services_to_migrate):
                service = self.services_to_migrate[self.current_service_idx]
                decided_service_id = getattr(service, 'id', None)
                
                # Execute action (migration)
                if self.is_valid_action(action):
                    if action < self.num_servers:
                        # Migrate to specified server
                        servers = EdgeServer.all()
                        if servers and action < len(servers):
                            target_server = servers[action]
                            try:
                                # Avoid migrating back to the same server; treat as no-op for both env and pattern
                                if service.server is not None and target_server == service.server:
                                    info['migration'] = False
                                    info['valid_action'] = True
                                    pattern_choice = -1
                                elif target_server.has_capacity_to_host(service):
                                    service.provision(target_server=target_server)
                                    info['migration'] = True
                                    info['valid_action'] = True
                                    pattern_choice = int(action)
                                else:
                                    info['valid_action'] = False
                                    pattern_choice = -1
                            except:
                                info['valid_action'] = False
                                pattern_choice = -1
                        else:
                            info['valid_action'] = False
                            pattern_choice = -1
                    else:
                        # No migration action
                        info['migration'] = False
                        info['valid_action'] = True
                        pattern_choice = -1
                else:
                    info['valid_action'] = False
                    pattern_choice = -1
                
                self.current_service_idx += 1
            
            # If we've processed all services (one pass), advance simulation time by one tick
            if self.current_service_idx >= len(self.services_to_migrate) or not self.services_to_migrate:
                # Treat lack of services as a valid no-op decision to avoid invalid-action penalties
                info['valid_action'] = True
                info['migration'] = False
                info['no_services'] = True
                try:
                    self.simulator.step()
                    self.current_step += 1
                    # Reset for next round
                    self.services_to_migrate = []
                    self.current_service_idx = 0
                except Exception as e:
                    # Handle step error gracefully
                    self.current_step += 1
            
            # Get new state
            state_after = self.get_state()
            
            # Add decision identifiers to info for evaluation pattern accounting
            if decided_service_id is not None:
                info['service_id'] = decided_service_id
                info['pattern_choice'] = pattern_choice
                info['no_migration_action'] = True if pattern_choice == -1 else False
            
            # Calculate reward
            reward = self.reward_calculator.calculate(
                state_before, action, state_after, info
            )
            
            # Add runtime metrics to info for logging/evaluation
            try:
                runtime_info = self.get_info()
                if isinstance(runtime_info, dict):
                    info.update(runtime_info)
            except:
                pass
            
            # Episode termination/truncation
            terminated = False
            truncated = self.current_step >= self.max_steps
            
            # Store current state
            self.last_state = state_after
            
            return state_after, reward, terminated, truncated, info
            
        except Exception as e:
            # Return safe defaults on error
            obs = self.get_state()
            return obs, 0.0, False, self.current_step >= self.max_steps, {'error': str(e)}
    
    def get_state(self) -> np.ndarray:
        """Get current environment state.
        
        Returns:
            State observation as numpy array
        """
        state = []
        
        try:
            servers = EdgeServer.all()
            
            if self.include_server_metrics and servers:
                # Add server metrics
                for server in servers:
                    try:
                        cpu_util = server.cpu_demand / server.cpu if server.cpu > 0 else 0
                        mem_util = server.memory_demand / server.memory if server.memory > 0 else 0
                        disk_util = server.disk_demand / server.disk if server.disk > 0 else 0
                        
                        # Get power consumption safely
                        try:
                            power = server.get_power_consumption()
                        except:
                            power = 0
                        
                        if self.normalize_state:
                            # Normalize utilization is already 0-1
                            # Normalize power (assuming max power from parameters)
                            if hasattr(server, 'power_model_parameters'):
                                max_power = server.power_model_parameters.get('max_power_consumption', 100)
                                power = power / max_power if max_power > 0 else 0
                        
                        state.extend([cpu_util, mem_util, disk_util, power])
                    except:
                        # Add zeros if server metrics fail
                        state.extend([0, 0, 0, 0])
            elif self.include_server_metrics:
                # No servers, add zeros
                state.extend([0] * (self.num_servers * 4))
            
            if self.include_service_metrics and self.current_service_idx < len(self.services_to_migrate):
                service = self.services_to_migrate[self.current_service_idx]
                
                # One-hot encoding for current server
                server_one_hot = [0] * self.num_servers
                if service.server and servers:
                    try:
                        server_idx = servers.index(service.server)
                        if 0 <= server_idx < self.num_servers:
                            server_one_hot[server_idx] = 1
                    except:
                        pass
                state.extend(server_one_hot)
                
                # Service resource demands (normalized if configured)
                try:
                    if self.normalize_state and servers:
                        max_cpu = max((s.cpu for s in servers), default=1)
                        max_mem = max((s.memory for s in servers), default=1)
                        cpu_demand = service.cpu_demand / max_cpu if max_cpu > 0 else 0
                        mem_demand = service.memory_demand / max_mem if max_mem > 0 else 0
                        state_size = service.state / 1000.0  # Normalize to MB scale
                    else:
                        cpu_demand = service.cpu_demand
                        mem_demand = service.memory_demand
                        state_size = service.state
                    
                    state.extend([cpu_demand, mem_demand, state_size])
                except:
                    state.extend([0, 0, 0])
                    
            elif self.include_service_metrics:
                # Pad with zeros if no service to migrate
                state.extend([0] * (self.num_servers + 3))
            
            # Ensure state has correct dimension
            if len(state) == 0:
                state = [0] * self.observation_space.shape[0]
            elif len(state) != self.observation_space.shape[0]:
                # Pad or truncate to match expected dimension
                if len(state) < self.observation_space.shape[0]:
                    state.extend([0] * (self.observation_space.shape[0] - len(state)))
                else:
                    state = state[:self.observation_space.shape[0]]
            
            return np.array(state, dtype=np.float32)
            
        except Exception as e:
            # Return zeros on error
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def is_valid_action(self, action: int) -> bool:
        """Check if action is valid in current state.
        
        Args:
            action: Server index or no-migration action
            
        Returns:
            True if action is valid
        """
        try:
            if action >= self.action_space.n:
                return False
            
            # If no services to migrate, treat any in-range action as a valid no-op
            if not self.services_to_migrate or self.current_service_idx >= len(self.services_to_migrate):
                return True
            
            # No-migration is always valid if enabled
            if self.allow_no_migration and action == self.num_servers:
                return True
            
            # Check if migration to server is valid
            servers = EdgeServer.all()
            if action < self.num_servers and action < len(servers) and self.current_service_idx < len(self.services_to_migrate):
                service = self.services_to_migrate[self.current_service_idx]
                target_server = servers[action]
                return target_server.has_capacity_to_host(service)
            
            return False
            
        except:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get additional environment information.
        
        Returns:
            Dictionary with environment metrics
        """
        try:
            servers = EdgeServer.all()
            services = Service.all()
            
            total_power = sum(server.get_power_consumption() for server in servers)
            num_migrations = len([s for s in services if s.being_provisioned])
            
            # Compute mean latency across users/applications if available
            mean_latency = 0.0
            total_latency = 0.0
            try:
                latencies = []
                for user in User.all():
                    # user.delays maps app.id (as str) -> delay
                    for app in getattr(user, 'applications', []) or []:
                        try:
                            d = user.delays.get(str(app.id), None)
                            if d is not None and not (isinstance(d, float) and (np.isinf(d) or np.isnan(d))):
                                latencies.append(float(d))
                        except Exception:
                            continue
                if latencies:
                    mean_latency = float(np.mean(latencies))
                    total_latency = float(np.sum(latencies))
            except Exception:
                pass
            
            info = {
                'total_power': total_power,
                'num_migrations': num_migrations,
                'current_step': self.current_step,
                'services_to_migrate': len(self.services_to_migrate),
                'mean_latency': mean_latency,
                'total_latency': total_latency,
            }
            
            return info
            
        except:
            return {
                'total_power': 0,
                'num_migrations': 0,
                'current_step': self.current_step,
                'services_to_migrate': 0,
                'mean_latency': 0.0,
                'total_latency': 0.0,
            }

    def get_current_service(self):
        """Return the current service under decision, if any.
        
        Used by baseline policies (e.g., Worst-Fit) that need to inspect the
        service before calling `step(action)`.
        """
        try:
            # Ensure we have a list of services to process, similar to logic in step()
            if not self.services_to_migrate:
                try:
                    self.services_to_migrate = [
                        s for s in Service.all() if not s.being_provisioned
                    ]
                    self.current_service_idx = 0
                except Exception:
                    self.services_to_migrate = []
                    self.current_service_idx = 0
            if 0 <= self.current_service_idx < len(self.services_to_migrate):
                return self.services_to_migrate[self.current_service_idx]
            return None
        except Exception:
            return None
