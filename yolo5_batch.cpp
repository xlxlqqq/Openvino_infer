#include <opencv2/opencv.hpp>
#include <cmath>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <openvino/openvino.hpp>
#include <stdlib.h>
#include <chrono>


const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

// 检测数据类（label， 置信度， 位置）
struct Detection {
	int class_id;
	float confidence;
	cv::Rect box;
};

struct Resize {
	cv::Mat resized_image;
	int dw;
	int dh;
};


class inferBatch {
private:
	std::string model_path_;
	std::string image_folder_;
	std::vector<std::string> image_paths_;
	int batch_size_;
	int input_W_;
	int input_H_;

	ov::Core core_;
	ov::CompiledModel compiled_model_;
	ov::InferRequest infer_request_;

	/// <summary>
	/// 用于一次读取多张图像,组成 B x 3 x W x H 的输入矩阵
	/// </summary>
	/// <param name="image_paths"></param>
	/// <returns></returns>
	std::vector<float> init_images_batch() {
		const auto& img_paths = image_paths_;
		const size_t batch_size = batch_size_;
		const int input_h = input_H_;
		const int input_w = input_W_;
		const int channels = 3;

		if (img_paths.size() < batch_size) {
			std::cout << "img_paths.size() < batch_size !!! Error" << std::endl;
			// TODO: 需要 padding 到模型支持的 batch_size
		}
		
		std::vector<float> blob_data(batch_size * channels * input_h * input_w);

		for (int b = 0; b < batch_size; ++b) {
			cv::Mat img = cv::imread(img_paths[b]);
			if (img.empty()) {
				std::cout << "img.empty() !!! Error" << std::endl;
			}

			cv::Mat resized;
			cv::resize(img, resized, cv::Size(input_w, input_h));
			cv::Mat float_img;
			resized.convertTo(float_img, CV_32F, 1.0 / 255.0);
			cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);

			std::vector<cv::Mat> chw;
			cv::split(float_img, chw);

			for (int c = 0; c < 3; ++c) {
				memcpy(blob_data.data() +
					b * 3 * input_h * input_w + c * input_h * input_w,
					chw[c].data,
					input_h * input_w * sizeof(float));
			}
		}
		return blob_data;
	}

	/// <summary>
	/// 推理结果解包裹为vector<vector<Detection>>
	/// </summary>
	/// <param name="output"></param>
	/// <param name="output_shape"></param>
	/// <param name="batch_detections"></param>
	/// <param name="conf_thresh"></param>
	void detect_batch(const float* output, const ov::Shape& output_shape,
		std::vector<std::vector<Detection>>& batch_detections,
		float conf_thresh = 0.25)
	{
		int batch_size = output_shape[0];        // 16
		int num_boxes = output_shape[1];         // 每张图的预测框数量
		int num_classes = output_shape[2] - 5;   // 4(mmbox)+1(obj_conf)+num_classes

		batch_detections.resize(batch_size);

		for (int b = 0; b < batch_size; ++b) {
			for (int i = 0; i < num_boxes; ++i) {
				const float* det = output + b * num_boxes * (num_classes + 5) + i * (num_classes + 5);
				float obj_conf = det[4];
				if (obj_conf < conf_thresh) continue;

				// 找最大类别置信度
				float max_cls_conf = 0.0f;
				int cls_id = -1;
				for (int c = 0; c < num_classes; ++c) {
					if (det[5 + c] > max_cls_conf) {
						max_cls_conf = det[5 + c];
						cls_id = c;
					}
				}

				float conf = obj_conf * max_cls_conf;
				if (conf < conf_thresh) continue;

				// xywh -> cv::Rect(x, y, w, h)
				float cx = det[0];
				float cy = det[1];
				float w = det[2];
				float h = det[3];
				int x = static_cast<int>(cx - w / 2.0f);
				int y = static_cast<int>(cy - h / 2.0f);
				int width = static_cast<int>(w);
				int height = static_cast<int>(h);

				Detection d;
				d.class_id = cls_id;
				d.confidence = conf;
				d.box = cv::Rect(x, y, width, height);

				batch_detections[b].push_back(d);
			}
		}
	}
	
	/// <summary>
	/// NMS（非极大抑制），将相似相近的检测框排除掉，只保留置信度最高的复选框
	/// </summary>
	/// <param name="batch_detections"></param>
	/// <param name="score_thresh"></param>
	/// <param name="nms_thresh"></param>
	void nms_batch_detections(
		std::vector<std::vector<Detection>>& batch_detections,
		float score_thresh = SCORE_THRESHOLD,
		float nms_thresh = NMS_THRESHOLD)
	{
		for (auto& detections : batch_detections) {
			std::vector<Detection> nms_result;

			// 按类别分别做 NMS（class-aware）
			std::map<int, std::vector<int>> class_map;
			for (int i = 0; i < detections.size(); ++i) {
				class_map[detections[i].class_id].push_back(i);
			}

			for (auto& cls : class_map) {
				std::vector<cv::Rect> boxes;
				std::vector<float> scores;
				std::vector<int> indices = cls.second;

				for (int idx : indices) {
					boxes.push_back(detections[idx].box);
					scores.push_back(detections[idx].confidence);
				}

				std::vector<int> keep;
				cv::dnn::NMSBoxes(boxes, scores, score_thresh, nms_thresh, keep);

				for (int k : keep) {
					nms_result.push_back(detections[indices[k]]);
				}
			}

			detections.swap(nms_result);
		}
	}


public:
	inferBatch(std::string model_path, std::string images_folder, int batch_size, int input_W, int input_H) :
		model_path_(model_path), image_folder_(images_folder), 
		batch_size_(batch_size), input_W_(input_W), input_H_(input_H) {
		
		core_.set_property("CPU", {
			ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
			ov::hint::num_requests(2)
		});
		compiled_model_ = core_.compile_model(model_path_, "CPU");
		infer_request_ = compiled_model_.create_infer_request();

		cv::glob(image_folder_, image_paths_, true);
	}
	
	~inferBatch() {
		
	}

	/// <summary>
	/// 外部调用接口
	/// </summary>
	/// <returns></returns>
	std::vector<std::vector<Detection>> infer() {
		std::vector<float> inputData = init_images_batch();

		// 创建输入 tensor
		ov::Tensor input_tensor(compiled_model_.input().get_element_type(),
			compiled_model_.input().get_shape(), inputData.data());

		infer_request_.set_input_tensor(input_tensor);

		// 推理
		infer_request_.infer();

		// 获取输出
		const ov::Tensor& output_tensor = infer_request_.get_output_tensor();
		float* detections = output_tensor.data<float>();
		ov::Shape output_shape = output_tensor.get_shape();

		// 后处理
		std::vector<std::vector<Detection>> batch_detections;
		detect_batch(detections, output_shape, batch_detections);

		// NMS非极大抑制
		nms_batch_detections(batch_detections);

		return batch_detections;
	}

};



int main()
{
	std::string modelPath = "./yolov5/model/best.xml";
	std::string imagePath = "./yolov5/images";
	int batchSize = 16;
	int inputW = 640;
	int inputH = 640;

	auto startTime = std::chrono::steady_clock::now();
	auto inferCore = inferBatch(modelPath, imagePath, batchSize, inputW, inputH);

	auto inferResult = inferCore.infer();

	auto endTime = std::chrono::steady_clock::now();
	std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << "ms" << std::endl;

	// test: 打印结果
	for (int i = 0; i < inferResult.size(); i++) {
		std::vector<Detection> detection_img = inferResult[i];
		std::cout << "image" << i << " detections: " << detection_img.size() << std::endl;
		for (int j = 0; j < detection_img.size(); j++) {
			Detection detection = detection_img[j];
			std::cout << detection.class_id << " " << detection.confidence << " " << detection.box.x << " " << detection.box.y << " " << detection.box.width << " " << detection.box.height << std::endl;
		}
	}

	system("pause");
	return 0;
}