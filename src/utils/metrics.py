"""Metrics collection and analysis utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


class MetricsCollector:
    """Collects and analyzes training metrics."""
    
    def __init__(self, log_dir: str = "logs/"):
        """Initialize metrics collector.
        
        Args:
            log_dir: Directory to save metrics
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_metrics = []
        self.step_metrics = []
        
    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        """Log episode-level metrics.
        
        Args:
            episode: Episode number
            metrics: Dictionary of metrics
        """
        metrics['episode'] = episode
        metrics['timestamp'] = pd.Timestamp.now()
        self.episode_metrics.append(metrics)
        
    def log_step(self, step: int, metrics: Dict[str, Any]):
        """Log step-level metrics.
        
        Args:
            step: Step number
            metrics: Dictionary of metrics
        """
        metrics['step'] = step
        self.step_metrics.append(metrics)
        
    def save_metrics(self):
        """Save collected metrics to disk."""
        # Save episode metrics
        if self.episode_metrics:
            df_episodes = pd.DataFrame(self.episode_metrics)
            df_episodes.to_csv(self.log_dir / "episode_metrics.csv", index=False)
            
        # Save step metrics
        if self.step_metrics:
            df_steps = pd.DataFrame(self.step_metrics)
            df_steps.to_csv(self.log_dir / "step_metrics.csv", index=False)
            
    def plot_training_curves(self, 
                            metrics_to_plot: List[str] = None,
                            save_path: Optional[str] = None):
        """Plot training curves.
        
        Args:
            metrics_to_plot: List of metric names to plot
            save_path: Path to save figure
        """
        if not self.episode_metrics:
            print("No episode metrics to plot")
            return
            
        df = pd.DataFrame(self.episode_metrics)
        
        if metrics_to_plot is None:
            # Plot all numeric columns except episode
            metrics_to_plot = [col for col in df.columns 
                              if df[col].dtype in ['float64', 'int64'] 
                              and col != 'episode']
        
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
            
        for ax, metric in zip(axes, metrics_to_plot):
            ax.plot(df['episode'], df[metric])
            ax.set_xlabel('Episode')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} over Training')
            ax.grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of training.
        
        Returns:
            Dictionary of summary statistics
        """
        if not self.episode_metrics:
            return {}
            
        df = pd.DataFrame(self.episode_metrics)
        
        stats = {}
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64'] and col != 'episode':
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'final': df[col].iloc[-1]
                }
                
        return stats


class PowerAnalyzer:
    """Analyzes power consumption patterns."""
    
    @staticmethod
    def analyze_server_utilization(edge_servers: List) -> Dict[str, float]:
        """Analyze server utilization metrics.
        
        Args:
            edge_servers: List of EdgeServer objects
            
        Returns:
            Dictionary of utilization metrics
        """
        cpu_utils = []
        mem_utils = []
        disk_utils = []
        
        for server in edge_servers:
            if server.cpu > 0:
                cpu_utils.append(server.cpu_demand / server.cpu)
            if server.memory > 0:
                mem_utils.append(server.memory_demand / server.memory)
            if server.disk > 0:
                disk_utils.append(server.disk_demand / server.disk)
        
        return {
            'avg_cpu_util': np.mean(cpu_utils) if cpu_utils else 0,
            'avg_mem_util': np.mean(mem_utils) if mem_utils else 0,
            'avg_disk_util': np.mean(disk_utils) if disk_utils else 0,
            'cpu_util_std': np.std(cpu_utils) if cpu_utils else 0,
            'mem_util_std': np.std(mem_utils) if mem_utils else 0,
            'disk_util_std': np.std(disk_utils) if disk_utils else 0,
        }
    
    @staticmethod
    def analyze_power_distribution(edge_servers: List) -> Dict[str, float]:
        """Analyze power consumption distribution.
        
        Args:
            edge_servers: List of EdgeServer objects
            
        Returns:
            Dictionary of power metrics
        """
        powers = [server.get_power_consumption() for server in edge_servers]
        
        return {
            'total_power': sum(powers),
            'avg_power': np.mean(powers) if powers else 0,
            'std_power': np.std(powers) if powers else 0,
            'min_power': min(powers) if powers else 0,
            'max_power': max(powers) if powers else 0,
            'active_servers': sum(1 for p in powers if p > 0),
        }


def create_training_report(metrics_collector: MetricsCollector,
                          save_path: str = "training_report.html"):
    """Create HTML training report.
    
    Args:
        metrics_collector: MetricsCollector with training data
        save_path: Path to save HTML report
    """
    stats = metrics_collector.get_summary_stats()
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EdgeSim-RL Training Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .metric-card { 
                background: #f9f9f9; 
                padding: 15px; 
                margin: 10px 0;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>EdgeSim-RL Training Report</h1>
        <h2>Summary Statistics</h2>
    """
    
    for metric_name, metric_stats in stats.items():
        html += f"""
        <div class="metric-card">
            <h3>{metric_name}</h3>
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
        """
        
        for stat_name, stat_value in metric_stats.items():
            html += f"<tr><td>{stat_name}</td><td>{stat_value:.4f}</td></tr>"
            
        html += """
            </table>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html)
        
    print(f"Training report saved to: {save_path}")