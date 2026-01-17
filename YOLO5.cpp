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

/*检测图像*/
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

/*绘制分选框，实现resize向原图像的转换*/
void print_result(std::vector<Detection> output, cv::Mat img, Resize res, std::string result_path)
{
	int index = result_path.find(".jpg");
	std::string sub_path = result_path.substr(0, index);
	cv::Mat cropped;
	int offset = 50;

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

		std::cout << detection.class_id << std::endl;
		cv::rectangle(img, cv::Point(box.x, box.y), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 3);
		cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(xmax, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
		cv::putText(img, std::to_string(classId), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 5, cv::Scalar(0, 0, 0));
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
	//compiled_model = core.compile_model(model, "AUTO");
	compiled_model = core.compile_model(model, "AUTO");
}


void main()
{
	std::string model_path = "./model/best.xml";
	std::vector<std::string> images_path;
	cv::glob("./input_images/*.bmp", images_path, true);

	ov::CompiledModel compiled_model;
	init_model(model_path, compiled_model);

	cv::Mat img;
	Resize res;
	ov::InferRequest infer_request;

	// 遍历路径下的每一张图像
	for (int i = 0; i < images_path.size(); i++)
	{
		// 将图像预处理至模型所需要条件下
		init_image(images_path[i], img, res, infer_request, compiled_model);

		const ov::Tensor& output_tensor = infer_request.get_output_tensor();
		ov::Shape output_shape = output_tensor.get_shape();
		float* detections = output_tensor.data<float>();

		std::vector<Detection> output; // 用以存储每一张图像的所有检测结果（一张图像可能检测到多个目标）
		detect(output_shape, detections, output);

		std::string result_path = "./output_images/" + std::to_string(i) + ".jpg";
		std::cout << result_path << std::endl;
		print_result(output, img, res, result_path);   // 将这张图像的所有检测结果保存
	}

	cv::waitKey(0);
	system("pause");
	return;
}