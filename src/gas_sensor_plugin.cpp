#include "icir_cleanroom/gas_sensor_plugin.hpp"

#include <regex>
#include <cmath>

namespace gazebo
{

GasSensorPlugin::GasSensorPlugin()
: total_detected_concentration(0.0)
{}

void GasSensorPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
    this->model = _model;

    if (!rclcpp::ok()) {
        rclcpp::init(0, nullptr);
    }

    if (_sdf->HasElement("detector_radius"))
        this->detector_radius = _sdf->Get<double>("detector_radius");
    else
        this->detector_radius = 0.1;

    std::string sensor_link_name = "gas_sensor_link";
    if (_sdf->HasElement("sensor_link_name"))
        sensor_link_name = _sdf->Get<std::string>("sensor_link_name");

    this->sensor_link = this->model->GetLink(sensor_link_name);
    if (!this->sensor_link) {
        gzerr << "[GasSensorPlugin] 센서 링크를 찾을 수 없습니다: " << sensor_link_name << std::endl;
        return;
    }

    std::string node_name = "gas_sensor_node_" + this->model->GetName();
    node_name = std::regex_replace(node_name, std::regex("[^a-zA-Z0-9_]"), "_");
    this->ros_node = std::make_shared<rclcpp::Node>(node_name);

    this->detected_pub =
        this->ros_node->create_publisher<std_msgs::msg::Float64>(
            "/gas_sensor/detected_concentration", 10);

    this->sensor_pose_pub =
        this->ros_node->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/gas_sensor/sensor_pose", rclcpp::QoS(10));

    this->FindAndSubscribeToSources();

    this->update_connection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&GasSensorPlugin::OnUpdate, this));
}

void GasSensorPlugin::SourceConcentrationCallback(
    const std_msgs::msg::Float64::SharedPtr msg,
    const std::string& source_name)
{
    this->source_concentrations[source_name] = msg->data;
}

void GasSensorPlugin::FindAndSubscribeToSources()
{
    physics::World* world = this->model->GetWorld().get();
    if (!world) return;

    auto model_count = world->ModelCount();
    std::regex source_regex(".*gas_source.*");

    for (unsigned int i = 0; i < model_count; ++i) {
        physics::ModelPtr m = world->ModelByIndex(i);
        if (!m) continue;

        std::string model_name = m->GetName();
        if (!std::regex_search(model_name, source_regex)) continue;

        if (this->source_subscribers.find(model_name) != this->source_subscribers.end())
            continue;

        std::string topic_name = "/" + model_name + "/concentration";

        auto callback = [this, model_name](const std_msgs::msg::Float64::SharedPtr msg) {
            this->SourceConcentrationCallback(msg, model_name);
        };

        auto sub = this->ros_node->create_subscription<std_msgs::msg::Float64>(
            topic_name, 10, callback);

        this->source_subscribers[model_name] = sub;
        this->source_models[model_name] = m;
        this->source_concentrations[model_name] = 0.0;
    }
}

double GasSensorPlugin::CalculateDetectedConcentration(double distance, double source_concentration)
{
    // 가우시안 확산 모델: 바람 없는 정적 환경에서의 가스 농도 분포
    // σ=1.5m 기준, 소스에서 멀어질수록 자연스럽게 감소
    constexpr double sigma = 1.5;
    double attenuation = std::exp(-(distance * distance) / (2.0 * sigma * sigma));
    return std::max(0.0, source_concentration * attenuation);
}

void GasSensorPlugin::OnUpdate()
{
    rclcpp::spin_some(this->ros_node);
    this->FindAndSubscribeToSources();

    this->total_detected_concentration = 0.0;

    ignition::math::Vector3d sensor_pos = this->sensor_link->WorldPose().Pos();
    ignition::math::Quaterniond sensor_rot = this->sensor_link->WorldPose().Rot();

    geometry_msgs::msg::PoseStamped sensor_pose_msg;
    sensor_pose_msg.header.stamp = this->ros_node->now();
    sensor_pose_msg.header.frame_id = "map";
    sensor_pose_msg.pose.position.x = sensor_pos.X();
    sensor_pose_msg.pose.position.y = sensor_pos.Y();
    sensor_pose_msg.pose.position.z = sensor_pos.Z();
    sensor_pose_msg.pose.orientation.x = sensor_rot.X();
    sensor_pose_msg.pose.orientation.y = sensor_rot.Y();
    sensor_pose_msg.pose.orientation.z = sensor_rot.Z();
    sensor_pose_msg.pose.orientation.w = sensor_rot.W();
    this->sensor_pose_pub->publish(sensor_pose_msg);

    for (const auto& source_pair : this->source_models) {
        const std::string& source_name = source_pair.first;
        physics::ModelPtr source_model = source_pair.second;
        if (!source_model || !source_model->GetWorld()) continue;

        double current_concentration = this->source_concentrations[source_name];
        if (current_concentration <= 0.0) continue;

        ignition::math::Vector3d source_pos = source_model->WorldPose().Pos();
        double distance = sensor_pos.Distance(source_pos);

        double detected = this->CalculateDetectedConcentration(distance, current_concentration);
        this->total_detected_concentration += detected;
    }

    auto msg = std_msgs::msg::Float64();
    msg.data = this->total_detected_concentration;
    this->detected_pub->publish(msg);
}

GZ_REGISTER_MODEL_PLUGIN(GasSensorPlugin)

}  // namespace gazebo
