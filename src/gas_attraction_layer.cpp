#include "icir_cleanroom/gas_attraction_layer.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>

namespace gas_layer
{
    // =============================================================
    // [ GasAttractionLocalLayer ]
    // =============================================================

    void GasAttractionLocalLayer::onInitialize()
    {
        current_ = true;
        enabled_ = true;

        auto node = node_.lock();
        if (!node) {
            RCLCPP_ERROR(rclcpp::get_logger("GasAttractionLocalLayer"), "node expired!");
            return;
        }

        loadMapAndIdwParams(node);
        initGrids();

        concentration_sub_ = node->create_subscription<std_msgs::msg::Float64>(
            "/gas_sensor/detected_concentration", rclcpp::QoS(1),
            std::bind(&GasAttractionLocalLayer::concentrationCallback, this, std::placeholders::_1));

        pose_sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/gas_sensor/sensor_pose", rclcpp::QoS(1),
            std::bind(&GasAttractionLocalLayer::poseCallback, this, std::placeholders::_1));

        reset_sub_ = node->create_subscription<std_msgs::msg::Empty>(
            "/gas_attraction_local_layer/reset", rclcpp::QoS(1),
            std::bind(&GasAttractionLocalLayer::resetCallback, this, std::placeholders::_1));

        value_pub_ = node->create_publisher<std_msgs::msg::Float32MultiArray>(
            "/gas_attraction_local_layer/value_grid", rclcpp::QoS(1));
        weight_pub_ = node->create_publisher<std_msgs::msg::Float32MultiArray>(
            "/gas_attraction_local_layer/weight_grid", rclcpp::QoS(1));
        last_publish_time_ = std::chrono::steady_clock::now();

        RCLCPP_INFO(rclcpp::get_logger("GasAttractionLocalLayer"), "loaded");
    }

    void GasAttractionLocalLayer::concentrationCallback(const std_msgs::msg::Float64::SharedPtr msg)
    {
        latest_concentration = msg->data;
        has_concentration_ = true;
        tryAccumulate();
    }

    void GasAttractionLocalLayer::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        latest_pose_ = *msg;
        has_pose_ = true;
        tryAccumulate();
    }

    // 정화 완료 등으로 그동안 누적된 농도 기억을 전부 지우고 미탐색(150) 상태로 되돌림
    void GasAttractionLocalLayer::resetCallback(const std_msgs::msg::Empty::SharedPtr /*msg*/)
    {
        initGrids();
        publishGrids();
        current_ = true;
        RCLCPP_INFO(rclcpp::get_logger("GasAttractionLocalLayer"), "grid reset");
    }

    void GasAttractionLocalLayer::tryAccumulate()
    {
        if (!has_pose_ || !has_concentration_) return;

        double sx = latest_pose_.pose.position.x;
        double sy = latest_pose_.pose.position.y;

        accumulateIntensityAt(sx, sy, latest_concentration);

        // global layer로의 publish는 5Hz로 제한 (센서 콜백은 ~1kHz라 그대로 보내면 부하가 큼)
        auto now = std::chrono::steady_clock::now();
        if (now - last_publish_time_ >= std::chrono::milliseconds(200)) {
            publishGrids();
            last_publish_time_ = now;
        }

        has_pose_ = false;
        has_concentration_ = false;
    }

    // 센서 위치(sensor_x, sensor_y)를 중심으로 radius_ 범위의 grid cell에
    // IDW 가중치로 concentration 값을 누적
    void GasAttractionLocalLayer::accumulateIntensityAt(double sensor_x, double sensor_y, double concentration)
    {
        int center_i = static_cast<int>((sensor_x - origin_x_) / resolution_);
        int center_j = static_cast<int>((sensor_y - origin_y_) / resolution_);
        int max_offset = static_cast<int>(radius_ / resolution_);

        for (int dx = -max_offset; dx <= max_offset; ++dx) {
            for (int dy = -max_offset; dy <= max_offset; ++dy) {
                int i = center_i + dx;
                int j = center_j + dy;
                if (i < 0 || i >= grid_width_ || j < 0 || j >= grid_height_) continue;

                double world_x = origin_x_ + i * resolution_;
                double world_y = origin_y_ + j * resolution_;
                double dist = std::hypot(world_x - sensor_x, world_y - sensor_y);
                if (dist > radius_) continue;

                double weight = std::exp(-(dist * dist) / (2.0 * sigma_ * sigma_));
                value_sum_grid_[j][i] += concentration * weight;
                weight_sum_grid_[j][i] += weight;
            }
        }

        current_ = true;
    }

    void GasAttractionLocalLayer::publishGrids()
    {
        std_msgs::msg::Float32MultiArray value_msg;
        std_msgs::msg::Float32MultiArray weight_msg;

        value_msg.layout.dim.resize(2);
        value_msg.layout.dim[0].label = "height";
        value_msg.layout.dim[0].size = grid_height_;
        value_msg.layout.dim[0].stride = grid_height_ * grid_width_;
        value_msg.layout.dim[1].label = "width";
        value_msg.layout.dim[1].size = grid_width_;
        value_msg.layout.dim[1].stride = grid_width_;
        weight_msg.layout = value_msg.layout;

        value_msg.data.reserve(grid_height_ * grid_width_);
        weight_msg.data.reserve(grid_height_ * grid_width_);

        for (int j = 0; j < grid_height_; ++j) {
            for (int i = 0; i < grid_width_; ++i) {
                value_msg.data.push_back(value_sum_grid_[j][i]);
                weight_msg.data.push_back(weight_sum_grid_[j][i]);
            }
        }

        value_pub_->publish(value_msg);
        weight_pub_->publish(weight_msg);
    }

    void GasAttractionLocalLayer::updateBounds(
        double robot_x, double robot_y, double /*robot_yaw*/,
        double *min_x, double *min_y, double *max_x, double *max_y
    ) {
        if (!isEnabled() || !current_) return;

        *min_x = std::min(*min_x, robot_x - radius_);
        *min_y = std::min(*min_y, robot_y - radius_);
        *max_x = std::max(*max_x, robot_x + radius_);
        *max_y = std::max(*max_y, robot_y + radius_);
    }

    void GasAttractionLocalLayer::updateCosts(
        nav2_costmap_2d::Costmap2D &master_grid, int min_i, int min_j, int max_i, int max_j
    ) {
        if (!isEnabled() || !current_) return;

        applyAttractionCosts(master_grid, min_i, min_j, max_i, max_j);

        current_ = false;
    }

    // =============================================================
    // [ GasAttractionGlobalLayer ]
    // =============================================================

    void GasAttractionGlobalLayer::onInitialize()
    {
        current_ = true;
        enabled_ = true;

        auto node = node_.lock();
        if (!node) {
            RCLCPP_ERROR(rclcpp::get_logger("GasAttractionGlobalLayer"), "node expired!");
            return;
        }

        loadMapAndIdwParams(node);
        initGrids();

        value_sub_ = node->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/gas_attraction_local_layer/value_grid", rclcpp::QoS(1),
            std::bind(&GasAttractionGlobalLayer::valueCallback, this, std::placeholders::_1));

        weight_sub_ = node->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/gas_attraction_local_layer/weight_grid", rclcpp::QoS(1),
            std::bind(&GasAttractionGlobalLayer::weightCallback, this, std::placeholders::_1));

        RCLCPP_INFO(rclcpp::get_logger("GasAttractionGlobalLayer"), "loaded");
    }

    void GasAttractionGlobalLayer::valueCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        const auto &data = msg->data;
        if (static_cast<int>(data.size()) != grid_width_ * grid_height_) return;

        // Local layer가 보낸 grid를 그대로 덮어씀
        // (Local layer 쪽이 측정 시점의 "현재" 값을 들고 있으므로, 매번 최신 상태로 받아옴)
        for (int j = 0; j < grid_height_; ++j)
            for (int i = 0; i < grid_width_; ++i)
                value_sum_grid_[j][i] = data[j * grid_width_ + i];

        current_ = true;
    }

    void GasAttractionGlobalLayer::weightCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        const auto &data = msg->data;
        if (static_cast<int>(data.size()) != grid_width_ * grid_height_) return;

        for (int j = 0; j < grid_height_; ++j)
            for (int i = 0; i < grid_width_; ++i)
                weight_sum_grid_[j][i] = data[j * grid_width_ + i];

        current_ = true;
    }

    void GasAttractionGlobalLayer::updateBounds(
        double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/,
        double *min_x, double *min_y, double *max_x, double *max_y
    ) {
        if (!isEnabled() || !current_) return;

        // global은 맵 전체에 대해 갱신
        *min_x = origin_x_;
        *min_y = origin_y_;
        *max_x = origin_x_ + grid_width_ * resolution_;
        *max_y = origin_y_ + grid_height_ * resolution_;
    }

    void GasAttractionGlobalLayer::updateCosts(
        nav2_costmap_2d::Costmap2D &master_grid, int min_i, int min_j, int max_i, int max_j
    ) {
        if (!isEnabled() || !current_) return;

        applyAttractionCosts(master_grid, min_i, min_j, max_i, max_j);

        current_ = false;
    }
} // namespace gas_layer

PLUGINLIB_EXPORT_CLASS(gas_layer::GasAttractionLocalLayer, nav2_costmap_2d::Layer)
PLUGINLIB_EXPORT_CLASS(gas_layer::GasAttractionGlobalLayer, nav2_costmap_2d::Layer)
