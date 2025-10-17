#!/usr/bin/env python
import argparse
import os
import time
import numpy as np
import torch

# ========= 解析命令行参数 ==========
parser = argparse.ArgumentParser(
    description="Potential Field Controller Demo in Isaac Lab (Repulsive Only)."
)
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# 这里添加 AppLauncher 相关的命令行参数
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ========= 启动 Omniverse 应用 ==========
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ========= 导入 Isaac Lab 相关模块 ==========
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip G1_MINIMAL_CFG
from isaaclab_assets import G1_MINIMAL_CFG  
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
# ========= 定义桌面场景配置 ==========
@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """桌面场景的配置"""
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "usd/g1_29dof_rev_1_0.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 1.0, 0.8),
            joint_pos={
                ".*_hip_pitch_joint": -0.20,
                ".*_knee_joint": 0.42,
                ".*_ankle_pitch_joint": -0.23,
                # ".*_elbow_pitch_joint": 0.87,
                "left_shoulder_roll_joint": 0.16,
                "left_shoulder_pitch_joint": 0.35,
                "right_shoulder_roll_joint": -0.16,
                "right_shoulder_pitch_joint": 0.35,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    "waist_pitch_joint",
                ],
                effort_limit=300,
                velocity_limit=100.0,
                stiffness={
                    ".*_hip_yaw_joint": 150.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 200.0,
                    ".*_knee_joint": 200.0,
                    "waist_yaw_joint": 200.0,
                    "waist_roll_joint": 200.0,
                    "waist_pitch_joint": 200.0,
                },
                damping={
                    ".*_hip_yaw_joint": 5.0,
                    ".*_hip_roll_joint": 5.0,
                    ".*_hip_pitch_joint": 5.0,
                    ".*_knee_joint": 5.0,
                    "waist_yaw_joint": 5.0,
                    "waist_roll_joint": 5.0,
                    "waist_pitch_joint": 5.0,
                },
                armature={
                    ".*_hip_.*": 0.01,
                    ".*_knee_joint": 0.01,
                    "waist_yaw_joint": 0.01,
                    "waist_roll_joint": 0.01,
                    "waist_pitch_joint": 0.01,
                },
            ),
            "feet": ImplicitActuatorCfg(
                effort_limit=20,
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                stiffness=20.0,
                damping=2.0,
                armature=0.01,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",

                ],
                effort_limit=300,
                velocity_limit=100.0,
                stiffness=40.0,
                damping=10.0,
                armature={
                    ".*_shoulder_.*": 0.01,
                    ".*_elbow_.*": 0.01,
                    ".*_wrist_.*": 0.01,

                },
            ),
        },
    )

# ========= 辅助函数：打印机器人资产下所有 prim 的路径 ==========
def list_robot_prims():
    try:
        # 使用 USD Python API 遍历 "/World/Robot" 下的所有子节点
        from pxr import Usd
        stage = Usd.Stage.GetCurrent()
        robot_prim = stage.GetPrimAtPath("/World/Robot")
        if robot_prim:
            print("在 '/World/Robot' 下发现的 prim 路径：")
            for prim in robot_prim.GetChildren():
                print(prim.GetPath())
        else:
            print("在 '/World/Robot' 未找到任何 prim，请检查机器人是否正确加载。")
    except Exception as e:
        print("获取 USD Stage 时出错：", e)

# ========= 运行仿真 ==========
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """主循环：运行仿真并打印出机器人关节与刚体信息（请先观察打印结果确定名称规则）"""
    
    # 先打印出机器人资产下所有 prim 路径，帮助你确认实际的命名规则
    list_robot_prims()
    
    # 如果你在上面的输出中找到了关节和身体对应的命名，可以在下面设置相应的正则表达式。
    # 例如，假设你观察后发现关节名称中包含 "joint" 而刚体名称中包含 "body"，可以这样设置：
    robot = scene["robot"]
    
    print("解析到的关节名称列表:", robot.joint_names)
    print("解析到的刚体名称列表:", robot.body_names)
    print(robot.data.joint_pos.shape)
    print(robot.data.joint_vel.shape)
    # self.robot.data.body_pos_w[:, self.ref_body_index],
    # self.robot.data.body_quat_w[:, self.ref_body_index],
    # self.robot.data.body_lin_vel_w[:, self.ref_body_index],
    # self.robot.data.body_ang_vel_w[:, self.ref_body_index],
    # self.robot.data.body_pos_w[:, self.key_body_indexes],

    step_count = 0
    while simulation_app.is_running():
        # 更新场景并执行一步物理仿真
        scene.update(dt=sim.get_physics_dt())
        scene.write_data_to_sim()
        sim.step()
        step_count += 1

# ========= 主函数 ==========
def main():
    # 创建仿真配置，指定仿真步长和设备（例如 "cuda:0" 或 "cpu"）
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # 初始化场景（由 InteractiveScene 管理所有实体）
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # 重置仿真环境
    sim.reset()
    print("[INFO]: Setup complete. Starting simulation...")

    # 进入主仿真循环
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
# 解析到的关节名称列表: ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
# 解析到的刚体名称列表: ['pelvis', 'imu_in_pelvis', 'left_hip_pitch_link', 'pelvis_contour_link', 'right_hip_pitch_link', 'waist_yaw_link', 'left_hip_roll_link', 'right_hip_roll_link', 'waist_roll_link', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_knee_link', 'right_knee_link', 'd435_link', 'head_link', 'imu_in_torso', 'left_shoulder_pitch_link', 'logo_link', 'mid360_link', 'right_shoulder_pitch_link', 'left_ankle_pitch_link', 'right_ankle_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_elbow_link', 'right_elbow_link', 'left_wrist_roll_link', 'right_wrist_roll_link', 'left_wrist_pitch_link', 'right_wrist_pitch_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link', 'left_rubber_hand', 'right_rubber_hand']
# torch.Size([1, 29])
# torch.Size([1, 29])