"""
Proper flywheel validation test that uses env.step() so the ActionManager
pipeline is correctly engaged (ImplicitActuator targets are properly set).

Action space: [6 leg joint_pos, 2 flywheel_vel]
flywheel_vel actions in [-1, 1] are scaled by 150.0 → [-150, 150] rad/s
"""
import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
import bipedal_locomotion

def main():
    env_cfg = parse_env_cfg("Isaac-Limx-SF-Blind-Flat-Play-v0", device=args_cli.device, num_envs=1)
    # Zero gravity so the robot floats
    env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    flywheel_joint_ids, flywheel_joint_names = robot.find_joints("flywheel_.*_Joint")
    print(f"\nFlywheel joints: {flywheel_joint_names} (ids: {flywheel_joint_ids})", flush=True)

    # Check critic shape
    obs, info = env.reset()
    critic_obs = obs.get("critic", None)
    if critic_obs is not None:
        print(f"Critic shape: {critic_obs.shape}", flush=True)

    # Lift robot off the ground
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = 1.5
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=base_env.device)  # identity quat
    root_state[:, 7:] = 0.0  # zero velocities
    robot.write_root_state_to_sim(root_state)

    # Construct action: [6 leg positions at 0, 2 flywheel velocities at MAX]
    # Action dim = 8: [abad_L, abad_R, hip_L, hip_R, knee_L, knee_R, flywheel_L, flywheel_R]
    action = torch.zeros(1, 8, device=base_env.device)
    action[:, 6] = 1.0   # flywheel_L at max (scaled to 150 rad/s)
    action[:, 7] = 1.0   # flywheel_R at max (scaled to 150 rad/s)

    print("\n--- PROPER FLYWHEEL TEST (using env.step) ---", flush=True)
    results = []
    for step in range(200):
        obs, rew, terminated, truncated, info = env.step(action)

        fw_vel_L = robot.data.joint_vel[0, flywheel_joint_ids[0]].item()
        fw_vel_R = robot.data.joint_vel[0, flywheel_joint_ids[1]].item()
        fw_rpm_L = fw_vel_L * 9.5493
        fw_rpm_R = fw_vel_R * 9.5493
        fw_tau_L = robot.data.applied_torque[0, flywheel_joint_ids[0]].item()
        pitch_rate = robot.data.root_ang_vel_w[0, 1].item()

        if step % 20 == 0 or step == 199:
            line = (f"Step {step:03d} | Torque: {fw_tau_L:6.1f} Nm | "
                    f"FW_L: {fw_vel_L:7.1f} rad/s ({fw_rpm_L:7.0f} RPM) | "
                    f"FW_R: {fw_vel_R:7.1f} rad/s ({fw_rpm_R:7.0f} RPM) | "
                    f"Torso Pitch Rate: {pitch_rate:7.2f} rad/s")
            print(line, flush=True)
            results.append(line)

        # If terminated or truncated, reset
        if terminated.any() or truncated.any():
            obs, info = env.reset()
            root_state = robot.data.default_root_state.clone()
            root_state[:, 2] = 1.5
            root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=base_env.device)
            root_state[:, 7:] = 0.0
            robot.write_root_state_to_sim(root_state)

    with open("proper_test_output.txt", "w") as f:
        f.write("\n".join(results))

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
