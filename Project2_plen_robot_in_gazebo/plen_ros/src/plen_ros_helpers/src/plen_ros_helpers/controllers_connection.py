#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import SwitchController

class ControllersConnection:
    def __init__(self, node: Node, namespace: str, controllers_list=None):
        self.node = node
        self.node.get_logger().warn("Start Init ControllersConnection")
        if controllers_list is None:
            controllers_list = ["joint_state_controller", "joint_trajectory_controller"]
        self.controllers_list = controllers_list
        self.switch_service_name = '/' + namespace + '/controller_manager/switch_controller'
        self.client = self.node.create_client(SwitchController, self.switch_service_name)
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error(f"Service {self.switch_service_name} not available!")
        else:
            self.node.get_logger().info(f"Found service {self.switch_service_name}")

    def switch_controllers(self, controllers_on, controllers_off, strictness=1):
        """
        Switch the specified controllers on and off.
        :param controllers_on: list of controller names to start.
        :param controllers_off: list of controller names to stop.
        :param strictness: strictness mode (e.g. 1 for BEST_EFFORT, 2 for STRICT).
        :return: True if successful, None otherwise.
        """
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error(f"Service {self.switch_service_name} not available!")
            return None

        request = SwitchController.Request()
        request.start_controllers = controllers_on
        request.stop_controllers = controllers_off  # corrected: use stop_controllers for controllers_off
        request.strictness = strictness

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result() is not None:
            return future.result().ok
        else:
            self.node.get_logger().error(f"{self.switch_service_name} service call failed: {future.exception()}")
            return None

    def reset_controllers(self):
        """
        Reset controllers by turning off then on the preconfigured list.
        :return: True if controllers were successfully reset, False otherwise.
        """
        reset_result = False

        result_off_ok = self.switch_controllers(controllers_on=[], controllers_off=self.controllers_list)
        if result_off_ok:
            result_on_ok = self.switch_controllers(controllers_on=self.controllers_list, controllers_off=[])
            if result_on_ok:
                reset_result = True
        return reset_result

