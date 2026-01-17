/// programmer: xlxlqqq
/// date: 20260116
/// function: 推理 unet 模型实现涂胶提取，（图像语义分割）

#include <iostream>
#include <cmath>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <openvino/openvino.hpp>

class UnetInfer {
public:
    UnetInfer(std::string input_image_path, std::string input_xml_path, std::string input_bin_path)
        : input_image_path_(input_image_path), input_xml_path_(input_xml_path), input_bin_path_(input_bin_path) {
        std::cout << "UnetInfer created" << std::endl;
    }

public:
    // 外部推理接口
    cv::Mat infer() {
        cv::Mat inputBlob = ImagePreProcess();
        auto infer_result = infer_core(inputBlob);
        cv::Mat mask = PostProcess(infer_result);

        return mask;
    }


private:
    std::string input_image_path_;
    std::string input_xml_path_;
    std::string input_bin_path_;
    //std::string encrypt_key_ = "0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF";

    // 输入图像预处理
    cv::Mat ImagePreProcess() {
        cv::Mat image = cv::imread(input_image_path_);
        if (image.empty()) {
            std::cerr << "无法读取输入图像" << std::endl;
            return cv::Mat{};
        }
        cv::resize(image, image, cv::Size(512, 512));
        image.convertTo(image, CV_32F, 1.0 / 255.0);

        // 2. 转换为NCHW格式
        cv::Mat inputBlob;
        cv::dnn::blobFromImage(image, inputBlob); // 1 * 3 * 512 * 512

        return inputBlob;
    }

    /// <summary>
    /// 用于一次读取多张图像,组成 B x 3 x W x H 的输入矩阵
    /// </summary>
    /// <param name="image_paths"></param>
    /// <returns></returns>
    cv::Mat ImagePreProcessBatch(const std::vector<std::string>& image_paths) {
        int batch = image_paths.size();
        std::vector<cv::Mat> images;

        for (auto& path : image_paths) {
            cv::Mat img = cv::imread(path);
            cv::resize(img, img, cv::Size(512, 512));
            img.convertTo(img, CV_32F, 1.0 / 255.0);
            images.push_back(img);
        }

        cv::Mat blob;
        cv::dnn::blobFromImages(images, blob);

        return blob;
    }

    // 推理结果绘制蒙版
    cv::Mat PostProcess(std::pair<ov::Tensor, ov::Shape> output) {

        ov::Shape output_shape = output.second;
        ov::Tensor output_tensor = output.first;

        float* output_data = output_tensor.data<float>();
        int output_size = output_shape[1] * output_shape[2] * output_shape[3];
        // 输出为 [2, 512, 512]
        int out_c = output_shape[1];
        int out_h = output_shape[2];
        int out_w = output_shape[3];

        // 取第1通道（如为前景概率）
        //cv::Mat prob_map(out_h, out_w, CV_32F, output_data + out_h * out_w); // 第2通道
        // 或者取最大概率通道（适用于多分类）
        //cv::Mat mask(out_h, out_w, CV_8U);
        //for (int y = 0; y < out_h; ++y) {
        //    for (int x = 0; x < out_w; ++x) {
        //        float max_val = -1e10;
        //        int max_idx = 0;
        //        for (int c = 0; c < out_c; ++c) {
        //            float val = output_data[c * out_h * out_w + y * out_w + x];
        //            if (val > max_val) {
        //                max_val = val;
        //                max_idx = c;
        //            }
        //        }
        //        mask.at<uchar>(y, x) = max_idx * 255; // 0或255
        //    }
        //}

        // 取第2通道作为前景概率
        cv::Mat prob_map(out_h, out_w, CV_32F, output_data + out_h * out_w);
        // 在这里取第一通道和第二通道，各自存成一张cv::mat

        cv::Mat prob_map_channel1(out_h, out_w, CV_32F, output_data); // 第1通道
        cv::Mat prob_map_channel2(out_h, out_w, CV_32F, output_data + out_h * out_w); // 第2通道

        cv::Mat mask(out_h, out_w, CV_8U);

        // 比较两个通道，如果第一个通道的数值大于第二个通道的数值，则对应位置的掩码设为255，否则设为0
        cv::compare(prob_map_channel1, prob_map_channel2, mask, cv::CMP_GT);


        // 阈值化得到掩码
        //cv::Mat mask;
        cv::threshold(prob_map, mask, 0.5, 255, cv::THRESH_BINARY);

        // 转为8位
        mask.convertTo(mask, CV_8U);

        return mask;
    }

    // 模型推理核心函数
    std::pair<ov::Tensor, ov::Shape> infer_core(cv::Mat inputBlob) {

        // 3. 加载OpenVINO模型
        ov::Core core;

        //ov::AnyMap cpu_config;
        //cpu_config["INFERENCE_NUM_THREADS"] = 8;      // 推理线程数
        //cpu_config["CPU_BIND_THREAD"] = true;         // 线程绑定
        //core.set_property("CPU", cpu_config);

        std::shared_ptr<ov::Model> model = core.read_model(input_xml_path_);
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
        //auto compiled_model = core.compile_model(model, "CPU", ov::hint::PerformanceMode::LATENCY);
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // 4. 设置输入
        auto input_port = compiled_model.input();
        ov::Tensor input_tensor(input_port.get_element_type(), { 1, 3, 512, 512 }, inputBlob.ptr<float>());
        infer_request.set_input_tensor(input_tensor);

        // 5. 推理
        infer_request.infer();

        // 6. 获取输出
        ov::Tensor output_tensor = infer_request.get_output_tensor();
        auto output_shape = infer_request.get_output_tensor().get_shape();
        //int output_size = output_shape[1] * output_shape[2] * output_shape[3];
        //std::cout << "output shape：" << output_shape[1] << " " << output_shape[2] << " " << output_shape[3] << std::endl;

        return { output_tensor, output_shape };
    }

};


int main() {

    std::string image_dir = "./unet/images_input"; // 批量图像目录
    // best_model_1: 更新后模型不合并BN
    // mixed_from_pytorch: 更新前模型合并BN
    // model: 更新前模型不合并BN
    // bn_folded_model: 更新后模型合并BN
    //std::string input_xml_path = "./unet/mixed_from_pytorch.xml";
    //std::string input_bin_path = "./unet/mixed_from_pytorch.bin";
    std::string input_xml_path = "./unet/model_3/best_model_static.xml";
    std::string input_bin_path = "./unet/model_3/best_model_static.bin";

    // 读取目录下所有图片
    std::vector<std::string> image_files;
    cv::glob(image_dir + "/*.jpg", image_files, false); // 可扩展 .png 等格式

    if (image_files.empty()) {
        std::cerr << "目录下没有找到图像！" << std::endl;
        return -1;
    }

    UnetInfer unet("", input_xml_path, input_bin_path); // 图片路径改为空，因为每次循环赋值

    double total_time_ms = 0.0;
    int count = 0;

    for (const auto& img_path : image_files) {
        if (count != 0) {
            //continue;
        }
        unet = UnetInfer(img_path, input_xml_path, input_bin_path); // 更新图片路径

        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat mask = unet.infer();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_time_ms += elapsed_ms;
        count++;

        // 保存每张图像结果
        std::string save_path = "./unet/output_mask_" + std::to_string(count) + ".png";
        cv::imwrite(save_path, mask);
        std::cout << "处理图像 " << img_path << " 用时: " << elapsed_ms << " ms, 保存: " << save_path << std::endl;
    }

    std::cout << "-------------------------" << std::endl;
    std::cout << "总计处理 " << count << " 张图像" << std::endl;
    std::cout << "平均推理时间: " << total_time_ms / count << " ms" << std::endl;

    return 0;
}