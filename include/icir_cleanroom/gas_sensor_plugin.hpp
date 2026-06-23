#ifndef GAS_SENSOR_PLUGIN_HPP
#define GAS_SENSOR_PLUGIN_HPP

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include "std_msgs/msg/float64.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <map>
#include <string>

namespace gazebo
{
    class GasSensorPlugin : public ModelPlugin
    {
        public:
            GasSensorPlugin();

            virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);
        private:
            void SourceConcentrationCallback(
                const std_msgs::msg::Float64::SharedPtr msg,
                const std::string &sourcE_name
            );

            // 소스 찾기/구독
            void FindAndSubscribeToSources();

            void OnUpdate();

            // 거리 기반 농도 계산
            double CalculateDetectedConcentration(
                double distance,
                double source_concentration
            );

            physics::ModelPtr model;
            physics::LinkPtr sensor_link;
            event::ConnectionPtr update_connection;
            std::shared_ptr<rclcpp::Node> ros_node;

            rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr detected_pub;
            rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr sensor_pose_pub; // 센서 위치 퍼블리셔

            std::map<std::string, rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr> source_subscribers;
            std::map<std::string, physics::ModelPtr> source_models;
            std::map<std::string, double> source_concentrations;

            double total_detected_concentration;
            double detector_radius;
    };
}

#endif