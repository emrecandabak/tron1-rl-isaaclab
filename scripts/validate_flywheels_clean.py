import argparse
import torch
from isaaclab.app import AppLauncher
import math

parser = argparse.ArgumentParser(description="Validate TRON1 Flywheels in IsaacLab.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
import bipedal_locomotion

def validate():
    env_cfg = parse_env_cfg("Isaac-Limx-SF-Blind-Flat-Play-v0", device=args_cli.device, num_envs=1)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    
    flywheel_joint_ids, flywheel_joint_names = robot.find_joints("flywheel_.*_Joint")
    
    env.reset()
    
    # We will manually hold the robot in the air by forcing its position to z=1.5
    # but allowing its orientation to change freely based on physics.
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = 1.5
    
    print("\n--- FLOATING MOMENTUM TEST ---")
    results = []
    for step in range(100):
        # Read current state so we don't overwrite orientation or velocities
        current_state = robot.data.root_state_w.clone()
        # Force position back to origin but keep orientation and velocities
        current_state[:, :3] = root_state[:, :3]
        robot.write_root_state_to_sim(current_state)
        
        # Command massive velocity to flywheels
        robot.set_joint_velocity_target(torch.tensor([[500.0, 500.0]], device=base_env.device), joint_ids=flywheel_joint_ids)
        
        base_env.scene.write_data_to_sim()
        base_env.sim.step()
        base_env.scene.update(dt=base_env.physics_dt)
        
        if step % 20 == 0 or step == 99:
            fw_vel = robot.data.joint_vel[0, flywheel_joint_ids[0]].item()
            fw_rpm = fw_vel * 9.5493
            fw_tau = robot.data.applied_torque[0, flywheel_joint_ids[0]].item()
            pitch_rate = robot.data.root_ang_vel_w[0, 1].item()
            line = f"Step {step:02d} | Torque: {fw_tau:5.1f} Nm | Flywheel: {fw_vel:6.1f} rad/s ({fw_rpm:6.0f} RPM) | Torso Pitch Rate: {pitch_rate:6.2f} rad/s"
            print(line, flush=True)
            results.append(line)

    with open("validation_output.txt", "w") as f:
        f.write("\n".join(results))
        
    env.close()

if __name__ == "__main__":
    validate()
    simulation_app.close()
