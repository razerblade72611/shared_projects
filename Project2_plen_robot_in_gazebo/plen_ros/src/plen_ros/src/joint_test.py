#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from gazebo_msgs.srv import SetModelState, SetModelConfiguration, DeleteModel, SpawnModel
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
from controller_manager_msgs.srv import LoadController, UnloadController, SwitchController
import numpy as np
import time

def call_service_sync(node, client, request):
    if not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().error(f"Service {client.srv_name} not available")
        return None
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    return future.result()

def pause(node, pause_client):
    req = Empty.Request()
    call_service_sync(node, pause_client, req)

def unpause(node, unpause_client):
    req = Empty.Request()
    call_service_sync(node, unpause_client, req)

def reset_model(node, reset_model_client, state):
    req = SetModelState.Request()
    req.model_state = state
    call_service_sync(node, reset_model_client, req)

def reset_joints(node, reset_joints_client, config):
    req = SetModelConfiguration.Request()
    req.model_name = config['model_name']
    req.urdf_param_name = config['urdf_param_name']
    req.joint_names = config['joint_names']
    req.joint_positions = config['joint_positions']
    call_service_sync(node, reset_joints_client, req)

def load_controller_service(node, load_client, controller_name):
    req = LoadController.Request()
    req.name = controller_name
    call_service_sync(node, load_client, req)

def unload_controller_service(node, unload_client, controller_name):
    req = UnloadController.Request()
    req.name = controller_name
    call_service_sync(node, unload_client, req)

def switch_controller_service(node, switch_client, start_controllers, stop_controllers, strictness):
    req = SwitchController.Request()
    req.start_controllers = start_controllers
    req.stop_controllers = stop_controllers
    req.strictness = strictness
    call_service_sync(node, switch_client, req)

def load_controllers(node, load_client, switch_client):
    node.get_logger().info("Loading controllers")
    controllers = [
        'joint_state_controller', '/plen/j1_pc', '/plen/j2_pc', '/plen/j3_pc',
        '/plen/j4_pc', '/plen/j5_pc', '/plen/j6_pc', '/plen/j7_pc',
        '/plen/j8_pc', '/plen/j9_pc', '/plen/j10_pc', '/plen/j11_pc',
        '/plen/j12_pc', '/plen/j13_pc', '/plen/j14_pc', '/plen/j15_pc',
        '/plen/j16_pc', '/plen/j17_pc', '/plen/j18_pc'
    ]
    for ctrl in controllers:
        load_controller_service(node, load_client, ctrl)
    switch_controller_service(node, switch_client, controllers, [], 2)

def unload_controllers(node, unload_client, switch_client):
    controllers = [
        'joint_state_controller', '/plen/j1_pc', '/plen/j2_pc', '/plen/j3_pc',
        '/plen/j4_pc', '/plen/j5_pc', '/plen/j6_pc', '/plen/j7_pc',
        '/plen/j8_pc', '/plen/j9_pc', '/plen/j10_pc', '/plen/j11_pc',
        '/plen/j12_pc', '/plen/j13_pc', '/plen/j14_pc', '/plen/j15_pc',
        '/plen/j16_pc', '/plen/j17_pc', '/plen/j18_pc'
    ]
    switch_controller_service(node, switch_client, [], controllers, 2)
    for ctrl in controllers:
        unload_controller_service(node, unload_client, ctrl)

def main():
    rclpy.init()
    node = rclpy.create_node('joint_test')

    # Create publishers for each joint command topic.
    rhip = node.create_publisher(Float64, '/plen/j1_pc/command', 1)
    rthigh = node.create_publisher(Float64, '/plen/j2_pc/command', 1)
    rknee = node.create_publisher(Float64, '/plen/j3_pc/command', 1)
    rshin = node.create_publisher(Float64, '/plen/j4_pc/command', 1)
    rankle = node.create_publisher(Float64, '/plen/j5_pc/command', 1)
    rfoot = node.create_publisher(Float64, '/plen/j6_pc/command', 1)
    lhip = node.create_publisher(Float64, '/plen/j7_pc/command', 1)
    lthigh = node.create_publisher(Float64, '/plen/8_pc/command', 1)  # (Preserving original topic name)
    lknee = node.create_publisher(Float64, '/plen/j9_pc/command', 1)
    lshin = node.create_publisher(Float64, '/plen/j10_pc/command', 1)
    lankle = node.create_publisher(Float64, '/plen/j11_pc/command', 1)
    lfoot = node.create_publisher(Float64, '/plen/j12_pc/command', 1)
    rshoulder = node.create_publisher(Float64, '/plen/j13_pc/command', 1)
    rarm = node.create_publisher(Float64, '/plen/j14_pc/command', 1)
    relbow = node.create_publisher(Float64, '/plen/j15_pc/command', 1)
    lshoulder = node.create_publisher(Float64, '/plen/j16_pc/command', 1)
    larm = node.create_publisher(Float64, '/plen/j17_pc/command', 1)
    lelbow = node.create_publisher(Float64, '/plen/j18_pc/command', 1)

    # Create service clients.
    delete_model_client = node.create_client(DeleteModel, 'gazebo/delete_model')
    spawn_model_client = node.create_client(SpawnModel, 'gazebo/spawn_urdf_model')
    reset_simulation_client = node.create_client(Empty, 'gazebo/reset_world')
    pause_client = node.create_client(Empty, 'gazebo/pause_physics')
    unpause_client = node.create_client(Empty, 'gazebo/unpause_physics')
    reset_model_client = node.create_client(SetModelState, 'gazebo/set_model_state')
    reset_joints_client = node.create_client(SetModelConfiguration, 'gazebo/set_model_configuration')
    load_controller_client = node.create_client(LoadController, 'plen/controller_manager/load_controller')
    switch_controller_client = node.create_client(SwitchController, 'plen/controller_manager/switch_controller')
    unload_controller_client = node.create_client(UnloadController, 'plen/controller_manager/unload_controller')

    # Wait for the necessary services.
    delete_model_client.wait_for_service()
    spawn_model_client.wait_for_service()
    reset_simulation_client.wait_for_service()
    pause_client.wait_for_service()
    unpause_client.wait_for_service()
    reset_model_client.wait_for_service()
    reset_joints_client.wait_for_service()
    load_controller_client.wait_for_service()
    switch_controller_client.wait_for_service()
    unload_controller_client.wait_for_service()

    node.get_logger().info("STARTED")

    # PAUSE the simulation.
    pause(node, pause_client)

    # Define a pose and reset the model state.
    pose = Pose()
    pose.position.x = 0.0
    pose.position.y = 0.0
    pose.position.z = 0.158
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0

    state = ModelState()
    state.model_name = "plen"
    state.pose = pose
    reset_model(node, reset_model_client, state)

    # Reset PLEN JOINTS.
    config = {
        'model_name': 'plen',
        'urdf_param_name': 'robot_description',
        'joint_names': [
            'rb_servo_r_hip', 'r_hip_r_thigh',
            'r_thigh_r_knee', 'r_knee_r_shin',
            'r_shin_r_ankle', 'r_ankle_r_foot',
            'lb_servo_l_hip', 'l_hip_l_thigh',
            'l_thigh_l_knee', 'l_knee_l_shin',
            'l_shin_l_ankle', 'l_ankle_l_foot',
            'torso_r_shoulder', 'r_shoulder_rs_servo',
            're_servo_r_elbow', 'torso_l_shoulder',
            'l_shoulder_ls_servo', 'le_servo_l_elbow'
        ],
        'joint_positions': [0.0] * 18
    }
    reset_joints(node, reset_joints_client, config)
    # Call reset_joints twice as in the original.
    reset_joints(node, reset_joints_client, config)

    # UNPAUSE the simulation.
    unpause(node, unpause_client)
    time.sleep(0.1)

    # Publish a command to the right and left arm position controllers.
    msg = Float64()
    msg.data = 1.57
    for i in range(1):  # Note: original comment indicates a higher iteration count may be needed.
        rarm.publish(msg)
        larm.publish(msg)

    # (Optional) Load or unload controllers as needed:
    # load_controllers(node, load_controller_client, switch_controller_client)
    # unload_controllers(node, unload_controller_client, switch_controller_client)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

