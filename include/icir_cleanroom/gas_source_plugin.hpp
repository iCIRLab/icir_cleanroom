#ifndef GAS_SOURCE_PLUGIN_HPP
#define GAS_SOURCE_PLUGIN_HPP

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/float64.hpp"
#include <rclcpp/logging.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

namespace gazebo
{
    class GasSourcePlugin : public ModelPlugin
    {
        public:
            GasSourcePlugin();

            virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);

        private:
            void PublishConcentration();
            void ConcentrationCallback(const std_msgs::msg::Float64::SharedPtr msg);
            void OnUpdate();

            physics::ModelPtr model;
            event::ConnectionPtr update_connection;
            std::shared_ptr<rclcpp::Node> ros_node;
            rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr concentration_pub;
            rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
            rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr concentration_sub;
            rclcpp::TimerBase::SharedPtr timer;

            double initial_concentration;   // 초기 가스 농도
            double radius;                  // 가스 발생원 반경
            
    };
}

#endif