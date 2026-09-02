"""
Script to inject dual pitch-axis flywheels on TRON1 USD model.
"""

import os
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Add flywheels to TRON1 USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# launch omniverse app to get pxr USD API with PhysX schema
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

def add_flywheels():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    orig_usd = os.path.join(project_root, "exts", "bipedal_locomotion", "bipedal_locomotion", "assets", "usd", "SF_TRON1A", "SF_TRON1A.usd")
    out_usd = os.path.join(project_root, "exts", "bipedal_locomotion", "bipedal_locomotion", "assets", "usd", "SF_TRON1A", "SF_TRON1A_flywheel.usd")

    print(f"\n==================================================")
    print(f"[USD BUILDER] Opening stage: {orig_usd}")
    stage = Usd.Stage.Open(orig_usd)
    if not stage:
        print("[USD BUILDER] ERROR: Failed to open stage!")
        return

    # Find the robot root and base_Link
    base_link_prim = None
    for prim in stage.Traverse():
        if prim.GetName() == "base_Link":
            base_link_prim = prim
            break
            
    if not base_link_prim:
        for prim in stage.Traverse():
            if "base" in prim.GetName().lower() and prim.IsA(UsdGeom.Xformable):
                print(f"Found candidate base prim: {prim.GetPath()}")
                base_link_prim = prim
                break

    if not base_link_prim:
        print("Could not find base_Link prim in stage!")
        return

    base_link_path = base_link_prim.GetPath()
    robot_root_path = base_link_path.GetParentPath()
    print(f"Base Link Path: {base_link_path}")
    print(f"Robot Root Path: {robot_root_path}")

    # We keep the full-size collision mesh so IsaacLab counts material properties (keeps Critic shape 270),
    # but explicitly set the physical inertia to match a 7.5cm flywheel!
    radius = 0.075
    length = 0.025
    mass = 0.6
    
    # Inertia: Iyy = 0.5 * m * r^2
    iyy = 0.5 * mass * (radius ** 2)
    ixx = 0.25 * mass * (radius ** 2) + (1.0 / 12.0) * mass * (length ** 2)
    izz = ixx

    z_offset = -0.15  # Moved 0.15m down from base center
    sides = [("L", 0.22), ("R", -0.22)]  # Expanded laterally to clear torso and leg housing
    
    for side_name, y_offset in sides:
        wheel_name = f"flywheel_{side_name}_Link"
        joint_name = f"flywheel_{side_name}_Joint"
        
        wheel_path = robot_root_path.AppendChild(wheel_name)
        joint_path = robot_root_path.AppendChild(joint_name)
        
        print(f"Creating Wheel: {wheel_path}")
        print(f"Creating Revolute Joint: {joint_path}")
        
        # 1. Define Cylinder Geometry / Rigid Body
        cylinder_geom = UsdGeom.Cylinder.Define(stage, wheel_path)
        cylinder_geom.CreateRadiusAttr(radius)
        cylinder_geom.CreateHeightAttr(length)
        cylinder_geom.CreateAxisAttr("Y")
        
        xform = cylinder_geom.AddTranslateOp()
        xform.Set(Gf.Vec3d(0.0, y_offset, z_offset))
        
        wheel_prim = stage.GetPrimAtPath(wheel_path)
        
        # Add Rigid Body API
        UsdPhysics.RigidBodyAPI.Apply(wheel_prim)
        
        # Add Mass API
        mass_api = UsdPhysics.MassAPI.Apply(wheel_prim)
        mass_api.CreateMassAttr(mass)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(ixx, iyy, izz))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        
        # Add Collision API so IsaacLab counts material properties (Critic shape 270).
        # The flywheel jam was caused by missing DriveAPI, not by collisions.
        collision_api = UsdPhysics.CollisionAPI.Apply(wheel_prim)
        
        # 2. Define Revolute Joint along Pitch (Y) Axis
        revolute_joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        revolute_joint.CreateAxisAttr("Y")
        
        revolute_joint.CreateBody0Rel().SetTargets([base_link_path])
        revolute_joint.CreateBody1Rel().SetTargets([wheel_path])
        
        revolute_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, y_offset, z_offset))
        revolute_joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        
        revolute_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        revolute_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        
        # CRITICAL: Add DriveAPI so ImplicitActuator can control the joint!
        # Without this, PhysX has no drive and ignores all velocity targets.
        # IsaacLab's ImplicitActuator will override stiffness/damping/maxForce at runtime.
        drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(joint_path), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)   # Will be overridden by ImplicitActuator
        drive.CreateDampingAttr(2.5)     # Will be overridden by ImplicitActuator
        drive.CreateMaxForceAttr(15.0)   # Will be overridden by ImplicitActuator

    print(f"Exporting modified stage to: {out_usd}")
    stage.Export(out_usd)
    print("Stage successfully exported!")

if __name__ == "__main__":
    add_flywheels()
    simulation_app.close()
