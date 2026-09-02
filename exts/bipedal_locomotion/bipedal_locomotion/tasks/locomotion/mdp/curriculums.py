from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def modify_event_parameter(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    value: Any | SceneEntityCfg,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies a parameter of an event at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        param_name: The name of the event term parameter.
        value: The new value for the event term parameter.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params[param_name] = value
        env.event_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def disable_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies the push velocity range at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the termination term.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        # Remove term settings
        term_cfg.params = dict()
        term_cfg.func = lambda env: torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env.termination_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1, device=env.device)
    return torch.zeros(1, device=env.device)


def velocity_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    reward_threshold: float = 0.70,
    max_lin_vel_x: tuple[float, float] = (-1.0, 2.5),
    max_lin_vel_y: tuple[float, float] = (-0.4, 0.4),
    step_size_x: float = 0.05,
    step_size_y: float = 0.02,
) -> torch.Tensor:
    """Progressively expand the velocity command range from walking to running as tracking performance improves."""
    command_term = env.command_manager.get_term(command_name)
    
    # Compute current velocity tracking accuracy directly
    asset = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    lin_vel_xy_err = torch.norm(command[:, :2] - asset.data.root_lin_vel_b[:, :2], dim=-1)
    tracking_score = torch.mean(torch.exp(-lin_vel_xy_err / 0.40)).item()
    
    current_min_x, current_max_x = command_term.cfg.ranges.lin_vel_x
    
    # When tracking score exceeds threshold, expand velocity limits towards running speeds
    if tracking_score > reward_threshold:
        new_max_x = min(current_max_x + step_size_x, max_lin_vel_x[1])
        new_min_x = max(current_min_x - step_size_x, max_lin_vel_x[0])
        command_term.cfg.ranges.lin_vel_x = (new_min_x, new_max_x)
        
        current_min_y, current_max_y = command_term.cfg.ranges.lin_vel_y
        new_max_y = min(current_max_y + step_size_y, max_lin_vel_y[1])
        new_min_y = max(current_min_y - step_size_y, max_lin_vel_y[0])
        command_term.cfg.ranges.lin_vel_y = (new_min_y, new_max_y)

    return torch.tensor(command_term.cfg.ranges.lin_vel_x[1], device=env.device)

