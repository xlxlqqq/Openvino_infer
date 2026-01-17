#include <opencv2/opencv.hpp>
#include <cmath>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <openvino/openvino.hpp>
#include <stdlib.h>


const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

// 检测数据类（label， 置信度， 位置）
struct Detection
{
	int class_id;
	float confidence;
	cv::Rect box;
};

struct Resize
{
	cv::Mat resized_image;
	int dw;
	int dh;
};

Resize resize_and_pad(cv::Mat& img, cv::Size new_shape) {
	float width = img.cols;
	float height = img.rows;
	float r = float(new_shape.width / std::max(width, height));
	int new_unpadW = int(round(width * r));
	int new_unpadH = int(round(height * r));
	Resize resize;
	cv::resize(img, resize.resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);

	resize.dw = new_shape.width - new_unpadW;
	resize.dh = new_shape.height - new_unpadH;
	cv::Scalar color = cv::Scalar(100, 100, 100);
	cv::copyMakeBorder(resize.resized_image, resize.resized_image, 0, resize.dh, 0, resize.dw, cv::BORDER_CONSTANT, color);

	return resize;
}

void init_image(std::string img_path, cv::Mat& img, Resize& res, ov::InferRequest& infer_request, ov::CompiledModel compiled_model)
{
	img = cv::imread(img_path);
	// resize image
	res = resize_and_pad(img, cv::Size(640, 640));

	// Step 5. Create tensor from image
	float* input_data = (float*)res.resized_image.data;
	ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);

	// Step 6. Create an infer request for model inference 
	infer_request = compiled_model.create_infer_request();
	infer_request.set_input_tensor(input_tensor);
	infer_request.infer();
}

/// <summary>
/// 用于一次读取多张图像,组成 B x 3 x W x H 的输入矩阵
/// </summary>
/// <param name="image_paths"></param>
/// <returns></returns>
float* init_images_batch(const std::vector<std::string>& img_paths) {
	const size_t batch_size = img_paths.size();
	const int input_h = 640;
	const int input_w = 640;
	const int channels = 3;

	float* blob_data = new float[batch_size * channels * input_h * input_w];

	for (int b = 0; b < batch_size; ++b) {
		cv::Mat img = cv::imread(img_paths[b]);

		cv::Mat resized;
		cv::resize(img, resized, cv::Size(input_w, input_h));
		cv::Mat float_img;
		resized.convertTo(float_img, CV_32F, 1.0 / 255.0);
		cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);

		std::vector<cv::Mat> chw;
		cv::split(float_img, chw);
		for (int c = 0; c < channels; ++c) {
			memcpy(blob_data + b * channels * input_h * input_w + c * input_h * input_w,
				chw[c].data, input_h * input_w * sizeof(float));
		}
	}
	return blob_data;
}

/*¼ì²âÍ¼Ïñ*/
void detect(ov::Shape output_shape, float* detections, std::vector<Detection>& output)
{
	// Step 8. Postprocessing including NMS  
	std::vector<cv::Rect> boxes;
	std::vector<int> class_ids;
	std::vector<float> confidences;

	for (int i = 0; i < output_shape[1]; i++) {
		float* detection = &detections[i * output_shape[2]];

		float confidence = detection[4];
		if (confidence >= CONFIDENCE_THRESHOLD) {
			float* classes_scores = &detection[5];
			cv::Mat scores(1, output_shape[2] - 5, CV_32FC1, classes_scores);
			cv::Point class_id;
			double max_class_score;
			cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

			if (max_class_score > SCORE_THRESHOLD) {
				confidences.push_back(confidence);
				class_ids.push_back(class_id.x);

				float x = detection[0];
				float y = detection[1];
				float w = detection[2];
				float h = detection[3];

				float xmin = x - (w / 2);
				float ymin = y - (h / 2);

				boxes.push_back(cv::Rect(xmin, ymin, w, h));
			}
		}
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

	for (int i = 0; i < nms_result.size(); i++)
	{
		Detection result;
		int idx = nms_result[i];
		result.class_id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
	}
}

/*»æÖÆ·ÖÑ¡¿ò£¬ÊµÏÖresizeÏòÔ­Í¼ÏñµÄ×ª»»*/
void print_result(std::vector<Detection> output, cv::Mat img, Resize res, std::string result_path)
{
	int index = result_path.find(".jpg");
	std::string sub_path = result_path.substr(0, index);
	cv::Mat cropped;
	int offset = 2;

	// Step 9. Print results and save Figure with detections
	for (int i = 0; i < output.size(); i++)
	{
		auto detection = output[i];
		auto box = detection.box;
		auto classId = detection.class_id;
		auto confidence = detection.confidence;
		float rx = (float)img.cols / (float)(res.resized_image.cols - res.dw);
		float ry = (float)img.rows / (float)(res.resized_image.rows - res.dh);

		box.x = rx * box.x;
		box.y = ry * box.y;
		box.width = rx * box.width;
		box.height = ry * box.height;

		box.x = std::max(0, box.x - offset);
		box.y = std::max(0, box.y - offset);
		box.width = std::min(box.width + 2 * offset, img.cols - box.x);
		box.height = std::min(box.height + 2 * offset, img.rows - box.y);

		//cout << box.x << "\t" << box.y << "\t" << box.width << "\t" << box.height << endl;

		cropped = img(box);
		std::string subsub = sub_path + "_" + std::to_string(i) + ".jpg";
		cv::imwrite(subsub, cropped);

		//cout << "Bbox" << i + 1 << ": Class: " << classId << " "
		//	<< "Confidence: " << confidence << " Scaled coords: [ "
		//	<< "cx: " << (float)(box.x + (box.width / 2)) / img.cols << ", "
		//	<< "cy: " << (float)(box.y + (box.height / 2)) / img.rows << ", "
		//	<< "w: " << (float)box.width / img.cols << ", "
		//	<< "h: " << (float)box.height / img.rows << " ]" << endl;

		float xmax = box.x + box.width;
		float ymax = box.y + box.height;

		// ¼ÆËãÖÐÐÄ×ø±ê
		float cx = box.x + box.width / 2.0f;
		float cy = box.y + box.height / 2.0f;

		// Êä³öÖÐÐÄ×ø±ê
		std::cout << "Detection " << i + 1 << " center: (cx: " << cx << ", cy: " << cy << ")" << std::endl;

		std::cout << detection.class_id << std::endl;
		cv::rectangle(img, cv::Point(box.x, box.y), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 3);
		cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(xmax, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
		cv::putText(img, std::to_string(classId), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 0));

		// ½«ÖÐÐÄ×ø±êÏÔÊ¾ÔÚÍ¼ÏñÉÏ
		std::string center_text = "Center: (" + std::to_string(int(cx)) + ", " + std::to_string(int(cy)) + ")";
		cv::putText(img, center_text, cv::Point(box.x, box.y + box.height + 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
	}
	cv::imwrite(result_path, img);
}

void init_model(std::string model_path, ov::CompiledModel& compiled_model)
{
	/*init model*/
	// Step 1. Initialize OpenVINO Runtime core
	ov::Core core;
	// Step 2. Read a model
	std::shared_ptr<ov::Model> model = core.read_model(model_path);
	// Step 4. Inizialize Preprocessing for the model
	ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
	// Specify input image format
	ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
	// Specify preprocess pipeline to input image without resizing
	ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255., 255., 255. });
	//  Specify model's input layout
	ppp.input().model().set_layout("NCHW");
	// Specify output results format
	ppp.output().tensor().set_element_type(ov::element::f32);
	// Embed above steps in the graph
	model = ppp.build();
	compiled_model = core.compile_model(model, "AUTO");
}

void detect_batch(const float* output, const ov::Shape& output_shape,
	std::vector<std::vector<Detection>>& batch_detections,
	float conf_thresh = 0.25)
{
	int batch_size = output_shape[0];        // 16
	int num_boxes = output_shape[1];         // 每张图的预测框数量
	int num_classes = output_shape[2] - 5;   // 4(box)+1(obj_conf)+num_classes

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


int main()
{
	std::string model_path = "./yolov5/model/best.xml";
	const std::string images_folder = "./yolov5/images/*.jpg";

	std::vector<std::string> images_path;
	cv::glob(images_folder, images_path, true);

	if (images_path.size() != 16) {
		std::cout << "Batch size must be 16" << std::endl;
		return -1;
	}

	ov::Core core;
	ov::CompiledModel compiled_model = core.compile_model(model_path, "CPU");
	ov::InferRequest infer_request = compiled_model.create_infer_request();

	float* inputData = init_images_batch(images_path);

	// 创建输入 tensor
	ov::Tensor input_tensor(compiled_model.input().get_element_type(),
		compiled_model.input().get_shape(), inputData);

	infer_request.set_input_tensor(input_tensor);

	// 推理
	infer_request.infer();

	// 获取输出
	const ov::Tensor& output_tensor = infer_request.get_output_tensor();
	float* detections = output_tensor.data<float>();
	ov::Shape output_shape = output_tensor.get_shape();

	// 后处理
	std::vector<std::vector<Detection>> batch_detections;
	detect_batch(detections, output_shape, batch_detections);

	// 打印位置图
	for (int i = 0; i < batch_detections.size(); i++) {
		std::vector<Detection> detection_img = batch_detections[i];
		std::cout << "image" << i << " detections: " << detection_img.size() << std::endl;
		for (int j = 0; j < detection_img.size(); j++) {
			Detection detection = detection_img[i];
			std::cout << detection.class_id << " " << detection.confidence << " " << detection.box.x << " " << detection.box.y << " " << detection.box.width << " " << detection.box.height << std::endl;
		}
	}

	delete[] inputData;
	system("pause");
	return 0;
}