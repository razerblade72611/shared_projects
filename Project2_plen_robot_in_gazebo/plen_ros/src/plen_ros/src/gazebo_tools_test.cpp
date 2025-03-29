#include <gazebo/gazebo.hh>
#include <gazebo/gazebo_client.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/transport/transport.hh>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/empty.hpp>  // For Empty service (if needed)
#include <functional>
#include "plen_ros/srv/iterate.hpp"  // ROS2 auto-generated header for the Iterate service

using Iterate = plen_ros::srv::Iterate;

// The callback now uses the ROS2 service signature.
void iterateCallback(const std::shared_ptr<rmw_request_id_t> /*request_header*/,
                     const std::shared_ptr<Iterate::Request> request,
                     std::shared_ptr<Iterate::Response> response,
                     gazebo::transport::PublisherPtr pub)
{
  // Create a Gazebo WorldControl message
  gazebo::msgs::WorldControl stepper;
  // Set the multi-step value from the service request
  stepper.set_multi_step(request->iterations);
  pub->Publish(stepper);

  response->result = true;
}

int main(int argc, char **argv)
{
  // Initialize Gazebo client (unchanged)
  gazebo::client::setup(argc, argv);
  gazebo::transport::NodePtr gz_node(new gazebo::transport::Node());
  gz_node->Init();

  // Initialize ROS2
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("gazebo_iterator");

  // Create a Gazebo transport publisher on the "~/world_control" topic
  gazebo::transport::PublisherPtr pub = gz_node->Advertise<gazebo::msgs::WorldControl>("~/world_control");
  pub->WaitForConnection();

  // Create a ROS2 service named "/iterate" using a lambda that captures the Gazebo publisher.
  auto service = node->create_service<Iterate>(
      "iterate",
      [pub](const std::shared_ptr<rmw_request_id_t> request_header,
            const std::shared_ptr<Iterate::Request> request,
            std::shared_ptr<Iterate::Response> response) -> void
      {
        iterateCallback(request_header, request, response, pub);
      });

  RCLCPP_INFO(node->get_logger(), "gazebo_iterator service is ready.");
  rclcpp::spin(node);

  rclcpp::shutdown();
  gazebo::client::shutdown();
  return 0;
}

