#include "icir_cleanroom/gas_source_plugin.hpp"
#include <regex>

namespace gazebo
{
    GasSourcePlugin::GasSourcePlugin() {}

    void GasSourcePlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
        model = _model;

        // SDF 파라미터 읽기
        if (_sdf->HasElement("initial_concentration"))
            initial_concentration = _sdf->Get<double>("initial_concentration");
        else
            initial_concentration = 100.0;

        if (_sdf->HasElement("radius"))
            radius = _sdf->Get<double>("radius");
        else
            radius = 0.1;

        // ROS 노드 초기화
        if (!rclcpp::ok()) {
            rclcpp::init(0, nullptr);
        }

        // 모델 이름으로 노드 이름 생성 (충돌 방지)
        std::string node_name = "gas_source_node_" + model->GetName();
        node_name = std::regex_replace(node_name, std::regex("[^a-zA-Z0-9_]"), "_");
        ros_node = std::make_shared<rclcpp::Node>(node_name);

        // 가스 농도를 발행할 publisher 생성
        std::string topic_name;
        if (model->GetName() == "gas_source") {
            topic_name = "gas_source/concentration";
        } else {
            topic_name = model->GetName() + "/concentration";
        }

        concentration_pub = ros_node->create_publisher<std_msgs::msg::Float64>(topic_name, 10);
        
        // DEBUG ONLY: 실제 환경에서는 존재하지 않는 위치 정보 발행
        pose_pub = ros_node->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/gas_source/pose", rclcpp::QoS(10)
        );

        // 외부에서 설정된 농도를 구독하는 subscription
        std::string set_topic;
        if (model->GetName() == "gas_source") {
            set_topic = "gas_source/concentration_set";
        } else {
            set_topic = model->GetName() + "/concentration_set";
        }

        concentration_sub = ros_node->create_subscription<std_msgs::msg::Float64>(
            set_topic, 10,
            std::bind(&GasSourcePlugin::ConcentrationCallback, this, std::placeholders::_1
            )
        );

        // 타이머 설정 (10hz로 정보 발행)
        timer = ros_node->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&GasSourcePlugin::PublishConcentration, this)
        );

        // Gazebo 업데이트 이벤트에 콜백 연결
        update_connection = event::Events::ConnectWorldUpdateBegin(
            std::bind(&GasSourcePlugin::OnUpdate, this)
        );
    }

    void GasSourcePlugin::ConcentrationCallback(const std_msgs::msg::Float64::SharedPtr msg)
    {
        initial_concentration = msg->data;
    }

    void GasSourcePlugin::OnUpdate()
    {
        rclcpp::spin_some(ros_node);
    }

    void GasSourcePlugin::PublishConcentration()
    {
        auto concentration_msg = std_msgs::msg::Float64();
        concentration_msg.data = initial_concentration;
        concentration_pub->publish(concentration_msg);

        // DEBUG ONLY: 디버그 시각화 노드에서만 사용, 실제 가스 센서 시스템에는 해당 토픽 없음
        auto pose_msg = geometry_msgs::msg::PoseStamped();
        pose_msg.header.stamp = ros_node->now();
        pose_msg.header.frame_id = "map";
        auto pos = model->WorldPose().Pos();
        pose_msg.pose.position.x = pos.X();
        pose_msg.pose.position.y = pos.Y();
        pose_msg.pose.position.z = pos.Z();
        pose_msg.pose.orientation.w = 1.0;
        pose_pub->publish(pose_msg);
    }

    GZ_REGISTER_MODEL_PLUGIN(GasSourcePlugin)
}