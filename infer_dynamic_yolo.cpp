#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>


std::vector<float> load_images_to_blob(
    const std::vector<std::string>& image_paths,
    int batch,
    int channels,
    int height,
    int width)
{
    std::vector<float> blob(batch * channels * height * width);

    for (int b = 0; b < batch; ++b) {
        cv::Mat img = cv::imread(image_paths[b]);
        if (img.empty()) {
            throw std::runtime_error("Failed to read image");
        }

        cv::resize(img, img, cv::Size(width, height));
        img.convertTo(img, CV_32F, 1.0 / 255.0);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    blob[
                        b * channels * height * width +
                            c * height * width +
                            h * width + w
                    ] = img.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
    }
    return blob;
}

int main() {
    try {
        const int BATCH = 16;
        const int C = 3;
        const int H = 640;
        const int W = 640;

        // 1. 初始化 OpenVINO
        ov::Core core;

        //core.set_property("GPU", {
        //    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)
        //    });

        // 2. 读取模型
        auto model = core.read_model("./yolov5/model/dynamic/best.xml");

        // 3. 设置输入 shape（动态 固定 16x3x640x640）
        ov::PartialShape input_shape{ BATCH, C, H, W };
        model->reshape({ {model->input().get_any_name(), input_shape} });

        // 4. 编译模型
        ov::CompiledModel compiled_model =
            core.compile_model(model, "GPU");

        ov::InferRequest infer_request =
            compiled_model.create_infer_request();

        // 5. 准备输入数据
        std::vector<std::string> image_paths;
        cv::glob("./yolov5/images/*.jpg", image_paths);
        image_paths.resize(BATCH);

        std::vector<float> input_blob =
            load_images_to_blob(image_paths, BATCH, C, H, W);

        // 6. 创建 Tensor
        ov::Tensor input_tensor(
            ov::element::f32,
            { BATCH, C, H, W },
            input_blob.data()
        );

        infer_request.set_input_tensor(input_tensor);

        // 7. 推理
        auto t0 = std::chrono::steady_clock::now();
        infer_request.infer();
        auto t1 = std::chrono::steady_clock::now();

        std::cout << "Inference time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
            << " ms" << std::endl;

        // 8. 取输出
        ov::Tensor output = infer_request.get_output_tensor();
        std::cout << "Output shape: " << output.get_shape() << std::endl;

        float* output_data = output.data<float>();

        // TODO: 后处理（YOLO decode / softmax / NMS）

    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
    }

    return 0;
}

