#ifndef GAS_ATTRACTION_LAYER_HPP
#define GAS_ATTRACTION_LAYER_HPP

#include <nav2_costmap_2d/costmap_layer.hpp>
#include <nav2_costmap_2d/layered_costmap.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <vector>
#include <algorithm>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include "ament_index_cpp/get_package_share_directory.hpp"

namespace gas_layer
{
    // [ 공통 ] Local/Global이 같은 좌표계(맵 전체 기준)의 grid를 쓰기 위한 베이스
    class GasAttractionBaseLayer : public nav2_costmap_2d::CostmapLayer
    {
        public:
            void reset() override {}

        protected:
            // TODO: 다음 단계에서 map_data.yaml 등으로 분리. 지금은 map/empty.yaml 값을 하드코딩.
            double resolution_ = 0.05;
            double origin_x_ = -6.53;
            double origin_y_ = -5.05;
            int grid_width_ = 262;
            int grid_height_ = 202;
            double max_concentration_ = 100.0;
            double sigma_ = 0.5; // IDW 가중치 폭

            std::vector<std::vector<float>> value_sum_grid_;
            std::vector<std::vector<float>> weight_sum_grid_;

            void initGrids()
            {
                value_sum_grid_ = std::vector<std::vector<float>>(grid_height_, std::vector<float>(grid_width_, 0.0f));
                weight_sum_grid_ = std::vector<std::vector<float>>(grid_height_, std::vector<float>(grid_width_, 0.0f));
            }

            // 농도가 높을수록 cost를 낮추는 공통 함수 (Local/Global 동일)
            void applyAttractionCosts(
                nav2_costmap_2d::Costmap2D &master_grid,
                int min_i, int min_j, int max_i, int max_j)
            {
                for (int i = min_i; i < max_i; ++i) {
                    for (int j = min_j; j < max_j; ++j) {
                        uint8_t existing_cost = master_grid.getCost(i, j);
                        if (existing_cost >= 253) continue;  // 물리 장애물은 건드리지 않음

                        double wx, wy;
                        master_grid.mapToWorld(i, j, wx, wy);

                        int vi = static_cast<int>((wx - origin_x_) / resolution_);
                        int vj = static_cast<int>((wy - origin_y_) / resolution_);
                        if (vi < 0 || vi >= grid_width_ || vj < 0 || vj >= grid_height_) continue;

                        double weight_sum = weight_sum_grid_[vj][vi];
                        if (weight_sum <= 1e-6) continue;

                        double avg_concentration = value_sum_grid_[vj][vi] / weight_sum;
                        double norm = std::clamp(avg_concentration / max_concentration_, 0.0, 1.0);
                        uint8_t cost = static_cast<uint8_t>((1.0 - norm) * 50.0);
                        cost = std::max<uint8_t>(cost, 1);  // FREE_SPACE(0)와 겹치지 않게 최소 1로 보정
                        master_grid.setCost(i, j, cost);
                    }
                }
            }

            void loadMapAndIdwParams(const rclcpp_lifecycle::LifecycleNode::SharedPtr &node)
            {
                node->declare_parameter(name_ + ".map_data_yaml", "");
                node->declare_parameter(name_ + ".idw_yaml", "");

                std::string map_yaml_path = node->get_parameter(name_ + ".map_data_yaml").as_string();
                std::string idw_yaml_path = node->get_parameter(name_ + ".idw_yaml").as_string();

                auto map_config = YAML::LoadFile(map_yaml_path)["map_data"];
                double map_width = map_config["map_width"].as<double>();
                double map_height = map_config["map_height"].as<double>();
                origin_x_ = map_config["map_origin_x"].as<double>();
                origin_y_ = map_config["map_origin_y"].as<double>();
                resolution_ = map_config["map_resolution"].as<double>();
                max_concentration_ = map_config["max_intensity"].as<double>();

                grid_width_ = static_cast<int>(map_width / resolution_);
                grid_height_ = static_cast<int>(map_height / resolution_);

                auto idw_config = YAML::LoadFile(idw_yaml_path)["idw"];
                sigma_ = idw_config["sigma"].as<double>();
            }
    };

    class GasAttractionLocalLayer : public GasAttractionBaseLayer
    {
        public:
            GasAttractionLocalLayer() = default;
            ~GasAttractionLocalLayer() = default;

            void onInitialize() override;
            void updateBounds(
                double robot_x, double robot_y, double robot_yaw,
                double *min_x, double *min_y, double *max_x, double *max_y
            ) override;
            void updateCosts(
                nav2_costmap_2d::Costmap2D &master_grid,
                int min_i, int min_j, int max_i, int max_j
            ) override;
            bool isClearable() override { return true; }
        private:
            double radius_ = 1.5;            // 측정 1회의 영향 반경

            geometry_msgs::msg::PoseStamped latest_pose_;
            double latest_concentration = 0.0;
            bool has_pose_ = false, has_concentration_ = false;

            rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr concentration_sub_;
            rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;

            rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr value_pub_;
            rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr weight_pub_;
            std::chrono::steady_clock::time_point last_publish_time_;

            void concentrationCallback(const std_msgs::msg::Float64::SharedPtr msg);
            void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
            void tryAccumulate();
            void accumulateIntensityAt(double sensor_x, double sensor_y, double concentration);
            void publishGrids();
    };

    class GasAttractionGlobalLayer : public GasAttractionBaseLayer
    {
        public:
            GasAttractionGlobalLayer() = default;
            ~GasAttractionGlobalLayer() = default;

            void onInitialize() override;
            void updateBounds(
                double robot_x, double robot_y, double robot_yaw,
                double *min_x, double *min_y, double *max_x, double *max_y
            ) override;
            void updateCosts(
                nav2_costmap_2d::Costmap2D &master_grid,
                int min_i, int min_j, int max_i, int max_j
            ) override;
            bool isClearable() override { return false; }
        private:
            rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr value_sub_;
            rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr weight_sub_;

            void valueCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
            void weightCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    };
} // namespace gas_layer
#endif