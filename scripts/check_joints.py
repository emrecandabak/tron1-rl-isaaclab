"""Check USD joint properties for all joints"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)

from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open("D:/Downloads/tron1-rl-isaaclab/exts/bipedal_locomotion/bipedal_locomotion/assets/usd/SF_TRON1A/SF_TRON1A_flywheel.usd")

results = []
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        name = prim.GetName()
        path = str(prim.GetPath())
        has_drive = prim.HasAPI(UsdPhysics.DriveAPI)
        
        # Check for drive
        drive_info = "NO DriveAPI"
        if has_drive:
            drive = UsdPhysics.DriveAPI(prim, "angular")
            stiffness = drive.GetStiffnessAttr().Get() if drive.GetStiffnessAttr() else "N/A"
            damping = drive.GetDampingAttr().Get() if drive.GetDampingAttr() else "N/A"
            max_force = drive.GetMaxForceAttr().Get() if drive.GetMaxForceAttr() else "N/A"
            drive_info = f"DriveAPI: stiffness={stiffness}, damping={damping}, max_force={max_force}"
        
        # Check limits
        rev = UsdPhysics.RevoluteJoint(prim)
        lower = rev.GetLowerLimitAttr().Get() if rev.GetLowerLimitAttr() else "N/A"
        upper = rev.GetUpperLimitAttr().Get() if rev.GetUpperLimitAttr() else "N/A"
        
        line = f"{name:30s} | {drive_info:60s} | limits=[{lower}, {upper}]"
        print(line, flush=True)
        results.append(line)

with open("joint_properties.txt", "w") as f:
    f.write("\n".join(results))

app_launcher.app.close()
