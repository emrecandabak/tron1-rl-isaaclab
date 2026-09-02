import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
import bipedal_locomotion
import torch

def main():
    env_cfg = parse_env_cfg('Isaac-Limx-SF-Blind-Flat-Play-v0', num_envs=1)
    # Set gravity to zero!
    env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    
    flywheel_joint_ids, _ = robot.find_joints("flywheel_.*_Joint")
    env.reset()
    
    # Lift it off the ground slightly so it doesn't touch the floor
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = 1.0
    robot.write_root_state_to_sim(root_state)
    base_env.scene.write_data_to_sim()
    base_env.sim.step()
    
    print("\n--- ZERO G MOMENTUM TEST ---")
    results = []
    for step in range(100):
        # Command 500 rad/s
        robot.set_joint_velocity_target(torch.tensor([[500.0, 500.0]], device=base_env.device), joint_ids=flywheel_joint_ids)
        
        base_env.scene.write_data_to_sim()
        base_env.sim.step()
        base_env.scene.update(dt=base_env.physics_dt)
        
        fw_vel = robot.data.joint_vel[0, flywheel_joint_ids[0]].item()
        fw_rpm = fw_vel * 9.5493
        fw_tau = robot.data.applied_torque[0, flywheel_joint_ids[0]].item()
        pitch_rate = robot.data.root_ang_vel_w[0, 1].item()
        
        if step % 10 == 0 or step == 99:
            line = f"Step {step:02d} | Torque: {fw_tau:5.1f} Nm | Flywheel: {fw_vel:6.1f} rad/s ({fw_rpm:6.0f} RPM) | Torso Pitch Rate: {pitch_rate:6.2f} rad/s"
            print(line, flush=True)
            results.append(line)

    with open("zerog_output.txt", "w") as f:
        f.write("\n".join(results))

main()
app_launcher.app.close()
