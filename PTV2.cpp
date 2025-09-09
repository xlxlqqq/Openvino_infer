/// programmer : xlxlqqq
/// date : 2025.03.31
/// 完成模型推理
/// 但是由于训练的很烂，推理结果也很烂，所以效果不好，结果仅供参考


#include <openvino/openvino.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <sstream>

/// <summary>
/// 读取 TXT 文件并随机采样 2024 个点
/// attention: 请确保 TXT 文件中每行数据为 4 维 (x, y, z, feature)
/// attention: 点云数据不宜比2048多太多，也不宜比2024少太多，否则可能导致推理结果不准确
/// </summary>
/// <param name="filename"></param>
/// <param name="target_points"></param>
/// <returns></returns>
std::vector<float> load_and_sample_pointcloud(const std::string& filename, size_t target_points = 2024) {
    std::vector<std::vector<float>> points;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("无法打开点云文件: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<float> point(4);  // 每个点 4 维 (x, y, z, feature)
        if (!(iss >> point[0] >> point[1] >> point[2] >> point[3])) {
            continue; // 跳过错误行
        }
        points.push_back(point);
    }

    file.close();
    size_t num_points = points.size();

    if (num_points == 0) {
        throw std::runtime_error("点云文件为空: " + filename);
    }

    // 进行随机采样
    std::vector<std::vector<float>> sampled_points(target_points);
    std::mt19937 gen(42);  // 设定随机种子，保证结果可复现
    std::uniform_int_distribution<size_t> dist(0, num_points - 1);

    for (size_t i = 0; i < target_points; ++i) {
        sampled_points[i] = points[dist(gen)];  // 随机选择点
    }

    // 展平为数组 (2024 * 4)
    std::vector<float> flattened_data;
    for (const auto& p : sampled_points) {
        flattened_data.insert(flattened_data.end(), p.begin(), p.end());
    }

    return flattened_data;
}


/// <summary>
/// 保存点云数据到 txt 文件
/// </summary>
/// <param name="point_cloud"></param>
/// <param name="num_points"></param>
/// <param name="num_features"></param>
/// <param name="filename"></param>
void save_point_cloud_to_txt(const std::vector<float>& point_cloud, size_t num_points, size_t num_features, const std::string& filename) {
    if (num_features != 3) {
        std::cerr << "错误: 点云数据必须是 3 维 (x, y, z)！" << std::endl;
        return;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开文件 " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            file << point_cloud[i * num_features + j] << " ";
        }
        file << "\n";  // 每个点换行
    }

    file.close();
    std::cout << "成功保存点云到 " << filename << std::endl;
}


/// <summary>
/// 计算 argmax 并转换类别索引
/// </summary>
/// <param name="logits"></param>
/// <param name="num_points"></param>
/// <returns></returns>
std::vector<int> compute_argmax(const std::vector<float>& logits, size_t num_points) {
    std::vector<int> labels(num_points);
    std::vector<float> label1Points;
    std::vector<float> label2Points;

    for (int i = 0; i < logits.size(); i++) {
        if (i % 2 == 0) {
            label1Points.push_back(logits[i]);
        }
        else {
            label2Points.push_back(logits[i]);
        }
    }
    for (size_t i = 0; i < num_points; ++i) {
        labels[i] = (logits[i * 2] > logits[i * 2 + 1]) ? 0 : 1; // 取概率较大的类别索引
        std::cout << labels[i] << std::endl;
    }
    return labels;
}


/// <summary>
/// 保存点云及其分割标签
/// </summary>
/// <param name="points"></param>
/// <param name="labels"></param>
/// <param name="num_points"></param>
/// <param name="filename"></param>
void save_point_cloud_with_labels(const std::vector<float>& points, const std::vector<int>& labels,
    size_t num_points, const std::string& filename) {
    if (points.size() != num_points * 3 || labels.size() != num_points) {
        std::cerr << "错误: 点云坐标或标签数据尺寸不匹配！" << std::endl;
        return;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开文件 " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < num_points; ++i) {
        file << points[i * 3] << " " << points[i * 3 + 1] << " " << points[i * 3 + 2]
            << " " << labels[i] << "\n";
    }

    file.close();
    std::cout << "分割后的点云已保存到: " << filename << std::endl;
}


int main() {
    try {
        // 1. 初始化 OpenVINO 核心对象
        ov::Core core;

        // 2. 读取模型（请替换为你的模型路径）
        std::string model_path = "./PTV2/inputmodels/model.xml";  // 确保 .xml 和 .bin 文件在同一目录
        std::shared_ptr<ov::Model> model = core.read_model(model_path);

        // 3. 编译模型（使用 CPU 推理）
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

        // 4. 创建推理请求
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // 5. 读取点云文件并进行随机采样
        std::string pointcloud_file = "./PTV2/test_after.txt";  // 替换为你的 TXT 文件路径
        std::vector<float> input_data = load_and_sample_pointcloud(pointcloud_file);

        // 6. 获取输入张量信息
        auto input_tensor = infer_request.get_input_tensor();
        ov::Shape input_shape = input_tensor.get_shape();  // 形状应为 (1, 2024, 4)
        size_t num_points = input_shape[1]; // 2048
        size_t num_features = input_shape[2]; // 4 (XYZ + intensity)

        // 7. 设置输入数据
        float* input_ptr = input_tensor.data<float>();
        std::copy(input_data.begin(), input_data.end(), input_ptr);

        // 8. 执行推理\


        infer_request.infer();

        // 9. 获取输出（支持多个输出）
        // PTv2的两个输出维度分别为 pointxyz(batch * 2048 * 3)、out(batch * 2048 * 2)
        size_t num_outputs = compiled_model.outputs().size();
        
        auto output_tensor_1 = infer_request.get_output_tensor(0);
        auto output_shape_1 = output_tensor_1.get_shape();  // 解析形状
        float* output_data_1 = output_tensor_1.data<float>();

        size_t batch_size_1 = output_shape_1[0];  // B
        size_t num_points_1 = output_shape_1[1]; // 2048
        size_t num_features_1 = output_shape_1[2]; // 3

        // 获取第二个输出 (B, 2048, 2)
        auto output_tensor_2 = infer_request.get_output_tensor(1);
        auto output_shape_2 = output_tensor_2.get_shape();
        float* output_data_2 = output_tensor_2.data<float>();

        size_t batch_size_2 = output_shape_2[0];
        size_t num_points_2 = output_shape_2[1];
        size_t num_features_2 = output_shape_2[2];

        // 解析输出数据
        std::vector<float> output_points(output_tensor_1.get_size());
        std::vector<float> output_logits(output_tensor_2.get_size());

        std::memcpy(output_points.data(), output_tensor_1.data<float>(), output_tensor_1.get_byte_size());
        std::memcpy(output_logits.data(), output_tensor_2.data<float>(), output_tensor_2.get_byte_size());

        // 10. 计算 argmax 获取最终分类标签
        std::vector<int> output_labels = compute_argmax(output_logits, num_points);

        //save_point_cloud_to_txt(output_points, num_points_1, num_features_1, "./PTV2/output_point_cloud.txt");

        // 11. 保存分割点云
        save_point_cloud_with_labels(output_points, output_labels, num_points, "./PTV2/segmented_point_cloud.txt");

        std::vector<float*> output_data(num_outputs);


    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
