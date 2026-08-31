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

    # --- MASSIVE PARALLEL SEARCH INITIALIZATION ---
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    ankle_indices, ankle_names = robot.find_joints("ankle_.*_Joint")
    print(f"\n[PASSIVE TUNER] Isolating ankles: {ankle_names}\n")

    num_envs = base_env.num_envs
    print(f"[PASSIVE TUNER] Generating {num_envs} random (Kp, Kd) combinations...")
    
    # Random Uniform Search (statistically better than Grid Search for N > 50)
    # Kp ranges from 10 to 100, Kd ranges from 0.1 to 4.0
    search_kp = torch.empty(num_envs, device=base_env.device).uniform_(10.0, 100.0)
    search_kd = torch.empty(num_envs, device=base_env.device).uniform_(0.1, 4.0)

    # Repeat for both left and right ankles: shape (num_envs, 2)
    stiffness_tensor = search_kp.unsqueeze(1).repeat(1, 2)
    damping_tensor = search_kd.unsqueeze(1).repeat(1, 2)

    # Write to simulation engine overrides
    robot.write_joint_stiffness_to_sim(stiffness_tensor, joint_ids=ankle_indices)
    robot.write_joint_damping_to_sim(damping_tensor, joint_ids=ankle_indices)
    
    print(f"[PASSIVE TUNER] Deployed combinations to all {num_envs} environments!")
    print(f"[PASSIVE TUNER] Neural Network actions for ankles are forced to 0.0.")
    
    total_test_steps = 1000 # Evaluate for 1000 steps (~20 seconds)
    print(f"[PASSIVE TUNER] Testing for {total_test_steps} steps. Only counting continuous forward movement...\n")

    current_streak = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
    max_streak = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
    
    timestep = 0
    # simulate environment
    while simulation_app.is_running() and timestep < total_test_steps:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            
            # --- FORCE PASSIVE ANKLE (Zero out NN action) ---
            actions[:, ankle_indices] = 0.0

            # env stepping
            obs, rew, dones, infos = env.step(actions)
            obs_history = infos["observations"].get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = infos["observations"].get("commands") 

            # --- TRACK FORWARD PROGRESS STREAKS ---
            # root_lin_vel_b is the base velocity in the base frame (x is forward)
            forward_vel = robot.data.root_lin_vel_b[:, 0]
            
            # Increment streak ONLY if it is moving forward at > 0.15 m/s
            moving_forward = forward_vel > 0.15
            current_streak += moving_forward.long()
            
            # Update max streak ever achieved by this environment
            max_streak = torch.max(max_streak, current_streak)
            
            # If an environment falls (dones), its continuous streak is broken!
            if dones.any():
                current_streak[dones] = 0
                
            timestep += 1
            if timestep % 100 == 0:
                print(f"  ... Step {timestep}/{total_test_steps} | Best streak so far: {max_streak.max().item()} steps")

    # Search concluded
    best_env = torch.argmax(max_streak).item()
    best_kp = search_kp[best_env].item()
    best_kd = search_kd[best_env].item()
    best_score = max_streak[best_env].item()

    print("\n" + "="*60)
    print(f"[PASSIVE TUNER] SEARCH COMPLETE!")
    if best_score == 0:
        print("[PASSIVE TUNER] ALL ROBOTS FAILED TO MOVE FORWARD.")
    else:
        print(f"[PASSIVE TUNER] Environment {best_env} achieved the longest forward-walking streak!")
        print(f"[PASSIVE TUNER] Forward Steps: {best_score} / {total_test_steps}")
        print(f"[PASSIVE TUNER] Optimal Stiffness (Kp): {best_kp:.2f}")
        print(f"[PASSIVE TUNER] Optimal Damping (Kd):   {best_kd:.2f}")
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
