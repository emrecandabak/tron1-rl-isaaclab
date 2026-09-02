"""
Check prims and joints in the newly generated USD
"""
import os
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics

def check_stage():
    usd_path = r"D:\Downloads\tron1-rl-isaaclab\exts\bipedal_locomotion\bipedal_locomotion\assets\usd\SF_TRON1A\SF_TRON1A_flywheel.usd"
    stage = Usd.Stage.Open(usd_path)
    print("\n--- STAGE CHECK ---")
    for prim in stage.Traverse():
        if "flywheel" in prim.GetName().lower():
            print(f"Found Prim: {prim.GetPath()} (Type: {prim.GetTypeName()})")
    print("-------------------\n")

if __name__ == "__main__":
    check_stage()
    simulation_app.close()
