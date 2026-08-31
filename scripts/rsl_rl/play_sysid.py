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
    
    # --- SYSID INITIALIZATION ---
    q_log = []
    q_dot_log = []
    tau_log = []
    q_def_log = []

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    # The TRON1 solefoot config names the ankles "ankle_L_Joint" and "ankle_R_Joint"
    ankle_indices, ankle_names = robot.find_joints("ankle_.*_Joint")
    print(f"\n[SYSID] Tracking ankles: {ankle_names}\n")

    num_steps_to_collect = 500
    timestep = 0
    # ----------------------------
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            # env stepping
            obs, _, _, infos = env.step(actions)
            obs_history = infos["observations"].get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = infos["observations"].get("commands") 

            # --- SYSID DATA COLLECTION ---
            if timestep < num_steps_to_collect:
                # Collect full data arrays (shape: [num_envs, num_ankles])
                q = robot.data.joint_pos[:, ankle_indices].cpu().numpy()
                q_dot = robot.data.joint_vel[:, ankle_indices].cpu().numpy()
                tau = robot.data.applied_torque[:, ankle_indices].cpu().numpy()
                q_def = robot.data.default_joint_pos[:, ankle_indices].cpu().numpy()

                q_log.append(q)
                q_dot_log.append(q_dot)
                tau_log.append(tau)
                q_def_log.append(q_def)

            elif timestep == num_steps_to_collect:
                print("\n[SYSID] Collection finished. Running regression...\n")

                # Flatten the full arrays for regression
                q_arr_full = np.concatenate(q_log, axis=0).flatten()
                q_dot_arr_full = np.concatenate(q_dot_log, axis=0).flatten()
                tau_arr_full = np.concatenate(tau_log, axis=0).flatten()
                q_def_arr_full = np.concatenate(q_def_log, axis=0).flatten()

                # Stance-Phase Isolation via Kinematics:
                # Only fit spring to data where torque > 2.0 (network doing work)
                stance_mask = np.abs(tau_arr_full) > 2.0

                q_arr = q_arr_full[stance_mask]
                q_dot_arr = q_dot_arr_full[stance_mask]
                tau_arr = tau_arr_full[stance_mask]
                q_def_arr = q_def_arr_full[stance_mask]

                if len(q_arr) == 0:
                    print("\n[SYSID] ERROR: No stance data collected. Torque threshold may be too high.\n")
                    break

                # For TRON1, default joint position is often exactly 0.0
                delta_q_arr = q_arr - q_def_arr
                X = np.stack([delta_q_arr, q_dot_arr], axis=1)
                y = tau_arr

                # Standard Least Squares
                coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

                Kp = -coef[0]
                Kd = -coef[1]

                print("=" * 40)
                print(f"Optimal Stiffness (Kp): {Kp:.4f}")
                print(f"Optimal Damping (Kd):   {Kd:.4f}")
                print(f"Resting Angle (from default): {np.mean(q_def_arr):.4f} rad")
                print("=" * 40)

                # --- VISUAL VERIFICATION PLOT ---
                try:
                    import matplotlib.pyplot as plt
                    # Extract data for Environment 0, Ankle 0 (Left Ankle) over time
                    tau_env0 = [t[0, 0] for t in tau_log]
                    q_env0 = [pos[0, 0] for pos in q_log]
                    q_dot_env0 = [vel[0, 0] for vel in q_dot_log]
                    q_def_env0 = [d[0, 0] for d in q_def_log]

                    # Calculate fitted passive spring
                    tau_fitted = [-Kp * (q - d) - Kd * qd for q, qd, d in zip(q_env0, q_dot_env0, q_def_env0)]

                    plt.figure(figsize=(10, 8))
                    plt.subplot(2, 1, 1)
                    plt.plot(tau_env0, label="Actual Neural Network Torque", linewidth=2)
                    plt.plot(tau_fitted, label="Fitted Passive Spring Torque", linestyle="--")
                    plt.title("Verification: Ankle Torque over 500 Steps (Env 0, Left Ankle)")
                    plt.ylabel("Torque (Nm)")
                    plt.legend()

                    plt.subplot(2, 1, 2)
                    plt.plot(q_env0, label="Ankle Position (rad)", color='orange')
                    plt.title("Verification: Ankle Joint Motion (Env 0)")
                    plt.xlabel("Timestep")
                    plt.ylabel("Position (rad)")
                    plt.legend()

                    plt.tight_layout()
                    plot_path = os.path.join(os.getcwd(), "sysid_verification_plot.png")
                    plt.savefig(plot_path)
                    print(f"\n[SYSID] SUCCESS: Visual verification plot saved to {plot_path}")
                except ImportError:
                    print("\n[SYSID] Matplotlib not installed, skipping verification plot.")

                break
            
            timestep += 1
            # -----------------------------

    # close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
