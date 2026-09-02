"""Check what get_material_properties returns"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)

import bipedal_locomotion
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv

env_cfg = parse_env_cfg('Isaac-Limx-SF-Blind-Flat-Play-v0', num_envs=1)
env = ManagerBasedRLEnv(cfg=env_cfg)
robot = env.unwrapped.scene['robot']

mat = robot.root_physx_view.get_material_properties()
print(f"\nmaterial_properties shape: {mat.shape}", flush=True)
print(f"num bodies: {robot.num_bodies}", flush=True)
print(f"body names: {robot.body_names}", flush=True)

with open("mat_props.txt", "w") as f:
    f.write(f"shape: {mat.shape}\n")
    f.write(f"num_bodies: {robot.num_bodies}\n")
    f.write(f"body_names: {robot.body_names}\n")
    f.write(f"values: {mat[0]}\n")

env.close()
app_launcher.app.close()
