"""Script to play a checkpoint if an RL agent from RSL-RL with SYSID."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import numpy as np
import time

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_mlp_as_onnx, export_policy_as_jit


def main():
    """Play with RSL-RL agent and run SYSID on the ankle joints."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    # specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor,
            export_model_dir,
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )
    # reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands")

    # --- SYSTEM IDENTIFICATION INITIALIZATION ---
    import matplotlib.pyplot as plt
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    
    # Get physical URDF indices to track the true physics state of the ankles
    phys_ankle_indices, phys_ankle_names = robot.find_joints("ankle_.*_Joint")
    print(f"\n[SYSID] Collecting data for: {phys_ankle_names}")
    
    total_test_steps = 500
    print(f"[SYSID] Recording active policy behavior for {total_test_steps} steps...\n")

    q_data = []
    v_data = []
    tau_data = []
    
    # For plotting robot 0
    plot_tau_l = []
    plot_tau_r = []

    timestep = 0
    # simulate environment
    while simulation_app.is_running() and timestep < total_test_steps:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            
            # env stepping (ACTIVE POLICY - No Zeroing!)
            obs, rew, dones, infos = env.step(actions)
            obs_history = infos["observations"].get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = infos["observations"].get("commands") 

            # Collect physical state of the ankles
            q = robot.data.joint_pos[:, phys_ankle_indices]
            v = robot.data.joint_vel[:, phys_ankle_indices]
            tau = robot.data.applied_torque[:, phys_ankle_indices]
            
            # Store for plotting (always tracking robot 0)
            plot_tau_l.append(tau[0, 0].item())
            plot_tau_r.append(tau[0, 1].item())
            
            # Filter to only capture data when the robot is actually standing/walking
            base_height = robot.data.root_pos_w[:, 2]
            valid_mask = (base_height > 0.65)
            
            if valid_mask.any():
                q_data.append(q[valid_mask].reshape(-1))
                v_data.append(v[valid_mask].reshape(-1))
                tau_data.append(tau[valid_mask].reshape(-1))
                
            timestep += 1
            if timestep % 100 == 0:
                print(f"  ... Collected {timestep}/{total_test_steps} steps of active control data")

    # --- PLOTTING ---
    print("\n[SYSID] Generating and saving ankle torque plot...")
    plt.figure(figsize=(10, 5))
    plt.plot(plot_tau_l, label="Left Ankle Torque (Nm)")
    plt.plot(plot_tau_r, label="Right Ankle Torque (Nm)", alpha=0.7)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    
    # Shade the 80 Nm limits
    plt.axhline(80, color='red', linestyle=':', alpha=0.5, label="Max Effort Limit")
    plt.axhline(-80, color='red', linestyle=':', alpha=0.5)
    
    plt.title("Active Policy Applied Ankle Torques over Time")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ankle_torque_plot.png")
    print("[SYSID] Plot saved to 'ankle_torque_plot.png'!")
    try:
        plt.show(block=False)
        plt.pause(3)
    except:
        pass

    # --- LEAST SQUARES REGRESSION ---
    print("\n[SYSID] Running Least Squares regression to extract spring parameters...")
    
    # Combine data
    Q = torch.cat(q_data).cpu()
    V = torch.cat(v_data).cpu()
    TAU = torch.cat(tau_data).cpu()
    
    # We want to fit: tau = -Kp * (q - q_rest) - Kd * v
    # This expands to: tau = -Kp * q - Kd * v + (Kp * q_rest)
    # Let C = (Kp * q_rest)
    # Regression model: tau = Beta1 * q + Beta2 * v + Beta3 * 1
    
    Y = TAU.unsqueeze(1)
    X = torch.stack([Q, V, torch.ones_like(Q)], dim=1)
    
    # Solve X * Beta = Y
    result = torch.linalg.lstsq(X, Y)
    beta = result.solution.squeeze()
    
    # Extract physical parameters from coefficients
    Kp = -beta[0].item()
    Kd = -beta[1].item()
    C = beta[2].item()
    
    q_rest = (C / Kp) if Kp != 0 else 0.0

    print("\n" + "="*60)
    print(f"[SYSID] SYSTEM IDENTIFICATION RESULTS")
    print(f"[SYSID] The active policy behaves similarly to a passive spring with:")
    print(f"        Stiffness (Kp) : {Kp:.2f}")
    print(f"        Damping (Kd)   : {Kd:.2f}")
    print(f"        Resting Angle  : {q_rest:.4f} rad")
    if Kp < 0:
        print(f"\n[SYSID] WARNING: Kp is NEGATIVE! This mathematically proves the active")
        print(f"                 policy is injecting energy (acting as an anti-spring).")
        print(f"                 A physical spring cannot replicate this behavior.")
    print("="*60 + "\n")

    # close the simulator
    env.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
