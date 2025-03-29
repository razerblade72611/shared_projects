#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetPhysicsProperties, SetModelConfiguration, SetModelState
from gazebo_msgs.msg import ODEPhysics, ModelState
from gazebo_msgs.srv import SetPhysicsProperties_Request as SetPhysicsPropertiesRequest  # alias for clarity
from gazebo_msgs.srv import SetModelConfiguration_Request as SetModelConfigurationRequest
from gazebo_msgs.srv import SetModelState_Request as SetModelStateRequest
from geometry_msgs.msg import Vector3
import numpy as np

class GazeboConnection:
    def __init__(self, node: Node, start_init_physics_parameters: bool, reset_world_or_sim: str, max_retry=20):
        self.node = node
        self._max_retry = max_retry

        # Create service clients.
        self.model_config_client = self.node.create_client(SetModelConfiguration, '/gazebo/set_model_configuration')
        self.unpause_client = self.node.create_client(Empty, '/gazebo/unpause_physics')
        self.pause_client = self.node.create_client(Empty, '/gazebo/pause_physics')
        self.reset_simulation_client = self.node.create_client(Empty, '/gazebo/reset_simulation')
        self.reset_world_client = self.node.create_client(Empty, '/gazebo/reset_world')
        self.set_physics_client = self.node.create_client(SetPhysicsProperties, '/gazebo/set_physics_properties')
        self.model_state_client = self.node.create_client(SetModelState, '/gazebo/set_model_state')

        # Wait for critical services.
        self.wait_for_services([
            self.model_config_client,
            self.unpause_client,
            self.pause_client,
            self.reset_simulation_client,
            self.reset_world_client,
            self.set_physics_client,
            self.model_state_client
        ])

        self.start_init_physics_parameters = start_init_physics_parameters
        self.reset_world_or_sim = reset_world_or_sim

        self.init_values()
        # Always pause the simulation after initialization.
        self.pauseSim()

    def wait_for_services(self, clients):
        for client in clients:
            if not client.wait_for_service(timeout_sec=5.0):
                self.node.get_logger().error(f"Service {client.srv_name} not available!")

    def call_service_sync(self, client, request):
        if not client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error(f"Service {client.srv_name} not available!")
            return None
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        return future.result()

    def teleport(self, pose, robot_name: str):
        request = SetModelStateRequest()
        request.model_state = ModelState()
        request.model_state.model_name = robot_name
        request.model_state.pose = pose
        # Zero out all twist components.
        request.model_state.twist.linear.x = 0.0
        request.model_state.twist.linear.y = 0.0
        request.model_state.twist.linear.z = 0.0
        request.model_state.twist.angular.x = 0.0
        request.model_state.twist.angular.y = 0.0
        request.model_state.twist.angular.z = 0.0
        request.model_state.reference_frame = 'world'
        self.call_service_sync(self.model_state_client, request)

    def reset_joints(self, joints_list, model_name: str):
        request = SetModelConfigurationRequest()
        request.model_name = model_name
        request.urdf_param_name = 'robot_description'
        request.joint_names = joints_list
        # Convert numpy zeros array to a Python list.
        request.joint_positions = np.zeros(len(request.joint_names)).tolist()
        self.call_service_sync(self.model_config_client, request)

    def pauseSim(self):
        self.node.get_logger().debug("Pausing simulation...")
        paused_done = False
        counter = 0
        while not paused_done and rclpy.ok():
            if counter < self._max_retry:
                try:
                    req = Empty.Request()
                    self.call_service_sync(self.pause_client, req)
                    paused_done = True
                except Exception as e:
                    counter += 1
                    self.node.get_logger().error("/gazebo/pause_physics service call failed")
            else:
                error_message = f"Maximum retries ({self._max_retry}) reached; please check Gazebo pause service"
                self.node.get_logger().error(error_message)
                raise Exception(error_message)

    def unpauseSim(self):
        self.node.get_logger().debug("Unpausing simulation...")
        unpaused_done = False
        counter = 0
        while not unpaused_done and rclpy.ok():
            if counter < self._max_retry:
                try:
                    req = Empty.Request()
                    self.call_service_sync(self.unpause_client, req)
                    unpaused_done = True
                except Exception as e:
                    counter += 1
                    self.node.get_logger().error(f"/gazebo/unpause_physics service call failed...Retrying {counter}")
            else:
                error_message = f"Maximum retries ({self._max_retry}) reached; please check Gazebo unpause service"
                self.node.get_logger().error(error_message)
                raise Exception(error_message)

    def resetSim(self):
        """
        Resets the simulation depending on the chosen reset option.
        """
        if self.reset_world_or_sim == "SIMULATION":
            self.resetSimulation()
        elif self.reset_world_or_sim == "WORLD":
            self.resetWorld()
        elif self.reset_world_or_sim == "NO_RESET_SIM":
            self.node.get_logger().error("NO RESET SIMULATION SELECTED")
        else:
            self.node.get_logger().error(f"WRONG Reset Option: {self.reset_world_or_sim}")

    def resetSimulation(self):
        if self.reset_simulation_client.wait_for_service(timeout_sec=5.0):
            try:
                req = Empty.Request()
                self.call_service_sync(self.reset_simulation_client, req)
            except Exception as e:
                self.node.get_logger().error("/gazebo/reset_simulation service call failed")
        else:
            self.node.get_logger().error("/gazebo/reset_simulation service not available")

    def resetWorld(self):
        if self.reset_world_client.wait_for_service(timeout_sec=5.0):
            try:
                req = Empty.Request()
                self.call_service_sync(self.reset_world_client, req)
            except Exception as e:
                self.node.get_logger().error("/gazebo/reset_world service call failed")
        else:
            self.node.get_logger().error("/gazebo/reset_world service not available")

    def init_values(self):
        self.resetSim()
        if self.start_init_physics_parameters:
            self.init_physics_parameters()
        else:
            self.node.get_logger().error("NOT Initialising Simulation Physics Parameters")

    def init_physics_parameters(self):
        """
        Initialize the simulation's physics parameters (e.g., gravity, friction).
        """
        self._time_step = 0.001
        self._max_update_rate = 0.0  # Setting to zero forces Gazebo to update as fast as possible

        self._gravity = Vector3()
        self._gravity.x = 0.0
        self._gravity.y = 0.0
        self._gravity.z = -9.81

        self._ode_config = ODEPhysics()
        self._ode_config.auto_disable_bodies = False
        self._ode_config.sor_pgs_precon_iters = 0
        self._ode_config.sor_pgs_iters = 50
        self._ode_config.sor_pgs_w = 1.3
        self._ode_config.sor_pgs_rms_error_tol = 0.0
        self._ode_config.contact_surface_layer = 0.001
        self._ode_config.contact_max_correcting_vel = 0.0
        self._ode_config.cfm = 0.0
        self._ode_config.erp = 0.2
        self._ode_config.max_contacts = 20

        self.update_gravity_call()

    def update_gravity_call(self):
        self.pauseSim()
        request = SetPhysicsPropertiesRequest()
        request.time_step = self._time_step
        request.max_update_rate = self._max_update_rate
        request.gravity = self._gravity
        request.ode_config = self._ode_config

        self.node.get_logger().debug(f"Updating gravity to: {request.gravity}")
        result = self.call_service_sync(self.set_physics_client, request)
        # Optionally, check result.success and result.status_message here.
        self.unpauseSim()

    def change_gravity(self, x, y, z):
        self._gravity.x = x
        self._gravity.y = y
        self._gravity.z = z
        self.update_gravity_call()

