import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import bipedal_locomotion
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv

env_cfg = parse_env_cfg('Isaac-Limx-SF-Blind-Flat-Play-v0', num_envs=1)
env = ManagerBasedRLEnv(cfg=env_cfg)
robot = env.unwrapped.scene['robot']

flywheel_body_ids, _ = robot.find_bodies('flywheel_.*_Link')
inertias = robot.root_physx_view.get_inertias()[0]
print('\n\n==== INERTIA OUTPUT ====')
print('Flywheel Inertias:', inertias[flywheel_body_ids])
print('========================\n\n')

simulation_app.close()
