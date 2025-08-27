#!/usr/bin/env python3
"""Test script to debug reward calculation."""

import sys
sys.path.append('src')

from envs.edge_env import EdgeEnv
import numpy as np

# Create environment with debug enabled
config = {
    "dataset_path": "datasets/sample_dataset.json",
    "reward": {
        "debug": True,  # Enable debug output
        "normalize": False  # Disable normalization to see raw values
    },
    "max_steps": 10
}

print("Initializing environment...")
env = EdgeEnv(config)

print("\nResetting environment...")
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Initial info: {info}")

print("\nTaking some steps...")
for i in range(5):
    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nStep {i+1}:")
    print(f"  Action: {action}")
    print(f"  Reward: {reward}")
    print(f"  Info: {info}")
    print(f"  Observation (first 5 values): {obs[:5]}")
    
    if terminated or truncated:
        break

print("\n\nChecking EdgeSimPy components...")
from edge_sim_py.components import EdgeServer, Service

servers = EdgeServer.all()
print(f"Number of servers: {len(servers)}")

if servers:
    for i, server in enumerate(servers[:3]):  # Check first 3 servers
        print(f"\nServer {i}:")
        print(f"  CPU: {server.cpu}, CPU demand: {server.cpu_demand}")
        print(f"  Memory: {server.memory}, Memory demand: {server.memory_demand}")
        
        # Check if server has power model
        try:
            power = server.get_power_consumption()
            print(f"  Power consumption: {power}")
        except Exception as e:
            print(f"  Error getting power: {e}")
        
        # Check power model parameters
        if hasattr(server, 'power_model_parameters'):
            print(f"  Power model params: {server.power_model_parameters}")
        else:
            print("  No power model parameters")

services = Service.all()
print(f"\nNumber of services: {len(services)}")
if services:
    for i, service in enumerate(services[:3]):  # Check first 3 services
        print(f"\nService {i}:")
        print(f"  CPU demand: {service.cpu_demand}")
        print(f"  Memory demand: {service.memory_demand}")
        print(f"  Server: {service.server}")
        print(f"  Being provisioned: {service.being_provisioned}")
