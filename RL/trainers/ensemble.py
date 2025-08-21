import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from tqdm import tqdm
import torch
from collections import deque

from ..agents.ensemble_agent import EnsembleAgent
from ..replay_buffers.replay_buffer import ReplayBuffer
from ..utils.metrics import calculate_metrics
from .walk_forward import WalkForwardTrainer

class EnsembleTrainer(WalkForwardTrainer):
    """
    Specialized trainer for ensemble agents
    Extends WalkForwardTrainer with ensemble-specific functionality
    """
    
    def __init__(self,
                 ensemble_agent: EnsembleAgent,
                 mdp,
                 config: Dict[str, Any]):
        
        super().__init__(ensemble_agent, mdp, config)
        
        # Ensemble-specific parameters
        self.base_agent_metrics = {name: [] for name in ensemble_agent.base_agents.keys()}
        self.meta_agent_metrics = []
        self.ensemble_predictions = []
        
        # Training schedules
        self.base_agent_schedule = config.get('base_agent_schedule', 'simultaneous')  # 'simultaneous', 'sequential', 'alternating'
        self.meta_training_delay = config.get('meta_training_delay', 1000)  # Steps before training meta-agent
        
    def _train_epoch(self, data: pd.DataFrame, epoch: int) -> Dict[str, float]:
        """Enhanced training for ensemble agents"""
        
        # Reset MDP if needed
        if hasattr(self.mdp, 'reset'):
            state = self.mdp.reset(data.iloc[:100])
        
        total_reward = 0
        episode_rewards = []
        base_losses = {name: [] for name in self.agent.base_agents.keys()}
        meta_losses = []
        ensemble_losses = []
        
        # Progress bar
        pbar = tqdm(range(100, len(data)), desc=f"Ensemble Epoch {epoch}")
        
        for idx in pbar:
            current_row = data.iloc[idx]
            
            # Get predictions from all base agents
            base_predictions = self.agent.get_base_predictions(state, deterministic=False)
            
            # Get ensemble action
            ensemble_action = self.agent.select_action(state, deterministic=False)
            
            # Step environment
            next_state, reward, done, info = self.mdp.step(ensemble_action, current_row)
            
            # Store transition for ensemble agent
            self.replay_buffer.add(state, ensemble_action, reward, next_state, done)
            
            # Store individual base agent experiences
            for name, base_agent in self.agent.base_agents.items():
                base_action = base_predictions[name]
                # Each base agent gets the same reward (could be modified)
                self.replay_buffer.add(state, base_action, reward, next_state, done)
            
            # Update agents based on schedule
            if len(self.replay_buffer) >= self.learning_starts and idx % self.update_frequency == 0:
                
                if self.base_agent_schedule == 'simultaneous':
                    # Update all agents simultaneously
                    for _ in range(self.gradient_steps):
                        batch = self.replay_buffer.sample(self.batch_size)
                        ensemble_loss_dict = self.agent.update(batch)
                        ensemble_losses.append(ensemble_loss_dict)
                        
                elif self.base_agent_schedule == 'sequential':
                    # Update base agents first, then meta-agent
                    for _ in range(self.gradient_steps):
                        batch = self.replay_buffer.sample(self.batch_size)
                        
                        # Update each base agent individually
                        for name, base_agent in self.agent.base_agents.items():
                            base_loss_dict = base_agent.update(batch)
                            base_losses[name].append(base_loss_dict)
                        
                        # Update meta-agent if enough steps
                        if idx > self.meta_training_delay:
                            meta_loss_dict = self.agent.meta_agent.update(batch)
                            meta_losses.append(meta_loss_dict)
                
                elif self.base_agent_schedule == 'alternating':
                    # Alternate between base agents and meta-agent
                    batch = self.replay_buffer.sample(self.batch_size)
                    
                    if idx % 2 == 0:
                        # Update base agents
                        for name, base_agent in self.agent.base_agents.items():
                            base_loss_dict = base_agent.update(batch)
                            base_losses[name].append(base_loss_dict)
                    else:
                        # Update meta-agent
                        if idx > self.meta_training_delay:
                            meta_loss_dict = self.agent.meta_agent.update(batch)
                            meta_losses.append(meta_loss_dict)
            
            # Store ensemble predictions for analysis
            self.ensemble_predictions.append({
                'step': idx,
                'base_predictions': base_predictions.copy(),
                'ensemble_action': ensemble_action,
                'actual_return': info.get('actual_return', 0),
                'reward': reward
            })
            
            # Track metrics
            total_reward += reward
            if done:
                episode_rewards.append(total_reward)
                total_reward = 0
                if hasattr(self.mdp, 'reset'):
                    state = self.mdp.reset(data.iloc[max(0, idx-100):idx])
            else:
                state = next_state
            
            # Update progress bar
            if len(ensemble_losses) > 0:
                avg_loss = np.mean([l.get('meta_critic_loss', l.get('critic_loss', 0)) for l in ensemble_losses[-10:]])
                pbar.set_postfix({
                    'reward': f'{reward:.4f}', 
                    'ensemble_loss': f'{avg_loss:.4f}',
                    'base_agents': len(self.agent.base_agents)
                })
        
        # Calculate epoch metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards) if episode_rewards else total_reward / len(data),
            'std_reward': np.std(episode_rewards) if episode_rewards else 0,
            'num_episodes': len(episode_rewards),
            'ensemble_agents_count': len(self.agent.base_agents)
        }
        
        # Add base agent metrics
        for name, losses in base_losses.items():
            if losses:
                metrics[f'{name}_mean_loss'] = np.mean([l.get('critic_loss', 0) for l in losses])
        
        # Add meta-agent metrics
        if meta_losses:
            metrics['meta_mean_loss'] = np.mean([l.get('critic_loss', 0) for l in meta_losses])
        
        # Add ensemble metrics
        if ensemble_losses:
            metrics['ensemble_mean_loss'] = np.mean([l.get('meta_critic_loss', l.get('critic_loss', 0)) for l in ensemble_losses])
        
        return metrics
    
    def _validate(self, data: pd.DataFrame) -> Tuple[Dict[str, float], List[Dict]]:
        """Enhanced validation with ensemble analysis"""
        
        predictions = []
        base_predictions_all = {name: [] for name in self.agent.base_agents.keys()}
        ensemble_predictions = []
        actual_returns = []
        
        # Reset MDP
        if hasattr(self.mdp, 'reset'):
            state = self.mdp.reset(data.iloc[:100])
        
        # Disable gradient computation
        with torch.no_grad():
            for idx in tqdm(range(100, len(data)), desc="Ensemble Validation"):
                current_row = data.iloc[idx]
                
                # Get predictions from all base agents
                base_preds = self.agent.get_base_predictions(state, deterministic=True)
                
                # Get ensemble prediction
                ensemble_action = self.agent.select_action(state, deterministic=True)
                
                # Step environment
                next_state, reward, done, info = self.mdp.step(ensemble_action, current_row)
                
                # Store all predictions
                pred_dict = {
                    'index': idx,
                    'ensemble_prediction': float(ensemble_action),
                    'actual_return': info['actual_return'],
                    'prediction_error': info['prediction_error']
                }
                
                # Add base agent predictions
                for name, pred in base_preds.items():
                    pred_dict[f'{name}_prediction'] = float(pred)
                    base_predictions_all[name].append(float(pred))
                
                predictions.append(pred_dict)
                ensemble_predictions.append(float(ensemble_action))
                actual_returns.append(info['actual_return'])
                
                # Update state
                if done and hasattr(self.mdp, 'reset'):
                    state = self.mdp.reset(data.iloc[max(0, idx-100):idx])
                else:
                    state = next_state
        
        # Calculate comprehensive validation metrics
        ensemble_metrics = calculate_metrics(
            np.array(ensemble_predictions),
            np.array(actual_returns)
        )
        
        # Add base agent metrics
        base_metrics = {}
        for name, base_preds in base_predictions_all.items():
            if base_preds:
                base_agent_metrics = calculate_metrics(
                    np.array(base_preds),
                    np.array(actual_returns)
                )
                for key, value in base_agent_metrics.items():
                    base_metrics[f'{name}_{key}'] = value
        
        # Combine metrics
        all_metrics = {**ensemble_metrics, **base_metrics}
        all_metrics['ensemble_vs_base_improvement'] = self._calculate_ensemble_improvement(
            ensemble_predictions, base_predictions_all, actual_returns
        )
        
        return all_metrics, predictions
    
    def _calculate_ensemble_improvement(self, ensemble_preds, base_preds_dict, actual_returns):
        """Calculate how much the ensemble improves over individual base agents"""
        
        ensemble_mae = np.mean(np.abs(np.array(ensemble_preds) - np.array(actual_returns)))
        
        base_maes = {}
        for name, preds in base_preds_dict.items():
            if preds:
                base_mae = np.mean(np.abs(np.array(preds) - np.array(actual_returns)))
                base_maes[name] = base_mae
        
        if base_maes:
            best_base_mae = min(base_maes.values())
            improvement = (best_base_mae - ensemble_mae) / best_base_mae
            return {
                'ensemble_mae': ensemble_mae,
                'best_base_mae': best_base_mae,
                'relative_improvement': improvement,
                'base_agent_maes': base_maes
            }
        
        return {'ensemble_mae': ensemble_mae}
    
    def get_ensemble_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of ensemble performance"""
        
        if not self.ensemble_predictions:
            return {}
        
        analysis = {
            'total_predictions': len(self.ensemble_predictions),
            'base_agents_count': len(self.agent.base_agents),
            'base_agent_names': list(self.agent.base_agents.keys())
        }
        
        # Analyze base agent agreement
        base_pred_arrays = {}
        for name in self.agent.base_agents.keys():
            base_pred_arrays[name] = [p['base_predictions'][name] for p in self.ensemble_predictions]
        
        # Calculate correlations between base agents
        correlations = {}
        names = list(base_pred_arrays.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                corr = np.corrcoef(base_pred_arrays[name1], base_pred_arrays[name2])[0, 1]
                correlations[f'{name1}_vs_{name2}'] = corr
        
        analysis['base_agent_correlations'] = correlations
        
        return analysis