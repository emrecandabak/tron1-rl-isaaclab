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
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.unwrapped.scene['robot']
    
    flywheel_body_ids, _ = robot.find_bodies('flywheel_.*_Link')
    inertias = robot.root_physx_view.get_inertias()[0]
    masses = robot.root_physx_view.get_masses()[0]
    
    print("\n" + "="*50)
    print("FLYWHEEL MASSES:", masses[flywheel_body_ids])
    print("FLYWHEEL INERTIAS (PhysX):", inertias[flywheel_body_ids])
    print("="*50 + "\n")
    
    # Save it to a file so we don't lose it if it crashes during close
    with open("inertia_out.txt", "w") as f:
        f.write(f"Masses: {masses[flywheel_body_ids].tolist()}\nInertias: {inertias[flywheel_body_ids].tolist()}\n")

main()
app_launcher.app.close()
