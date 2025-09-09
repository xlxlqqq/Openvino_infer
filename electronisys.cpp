// 功能：使用OPENVINO推理一个最简单的分类模型
// 项目：电泳工件分类
// programmer: xlxlqqq
// date: 2024.08.17
// version: 1.0
// 状态：完成

//#include <iostream>
//
//#include <string>
//#include <filesystem>
//#include <io.h>
//
//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//
//#include <openvino/openvino.hpp>

#include <openvino_classification.h>

//class openvino_classification
//{
//	public:
//
//		/// function: check if a folder exists
//		/// return: true if exists, false if not
//		/// programmer: xlxlqqq
//		/// date: 2024.08.17
//		/// version: 1.0
//		bool folder_exists(std::string folder_path)
//		{
//			if (_access(folder_path.c_str(), 0) == -1)
//			{
//				return false;
//			}
//			return true;
//		}
//
//		/// @function: test openvino classification model with one input image 
//		/// @return: 0 if success, -1 if failed 
//		/// @programmer: xlxlqqq 
//		/// @date: 2024.08.17 
//		/// @version: 1.0 
//		int infer_one_image(std::string input_image_path, std::string model_path, int& class_index)
//		{
//			cv::Mat img = cv::imread(input_image_path);
//			if (img.empty())
//			{
//				std::cerr << "Can't open or find the image!\n";
//				return -1;
//			}
//			cv::Mat resized_image;
//			cv::resize(img, resized_image, cv::Size(224, 224));
//			resized_image.convertTo(resized_image, CV_32F);
//			resized_image = resized_image / 255.0;
//
//			std::vector<cv::Mat> channels(3);
//			cv::split(resized_image, channels);
//			cv::Mat chw_image(224, 224, CV_32FC3);
//			for (int c = 0; c < 3; ++c)
//			{
//				cv::Mat channel(224, 224, CV_32F, chw_image.ptr < float >(c));
//				resized_image(cv::Rect(0, 0, 224, 224)).copyTo(channel);
//			}
//
//			ov::Core core;
//			std::shared_ptr<ov::Model> model = core.read_model("./electronisys/models/model.xml");
//
//			ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
//			ov::InferRequest infer_request = compiled_model.create_infer_request();
//
//			auto input_tensor = infer_request.get_input_tensor();
//			float* input_data = input_tensor.data<float>();
//			std::memcpy(input_data, chw_image.datastart, chw_image.total() * chw_image.elemSize());
//
//			infer_request.infer();
//			auto output_tensor = infer_request.get_output_tensor();
//			const float* output_data = output_tensor.data<const float>();
//
//			std::cout << "Inferrence result: ";
//			int max_index = -1;
//			float max_value = -1.0;
//			for (size_t k = 0; k < output_tensor.get_size(); k++)
//			{
//				if (output_data[k] > max_value)
//				{
//					max_value = output_data[k];
//					max_index = k;
//				}
//			}
//			if (max_index == -1)
//			{
//				std::cout << " ERROR when inference!!!" << std::endl;
//				return -1;
//			}
//			class_index = max_index + 1;
//			std::cout << "Image named as " << input_image_path << " is classified as the " << max_index + 1 << " class with probability " << endl;
//
//			return 0;
//		}
//
//		/// function: get all images in a folder(png, jpg ,jpeg, bmp)
//		/// return: 0 if success, -1 if failed
//		/// programmer: xlxlqqq
//		/// date: 2024.08.17
//		/// version: 1.0
//		int get_all_images_in_folder(std::string input_folder_path, std::vector<std::string>& images_path)
//		{
//			if (!folder_exists(input_folder_path))
//			{
//				std::cout << "Folder not exists!" << endl;
//				return -1;
//			}
//			std::vector<std::string> all_types;
//			all_types.push_back(".png");
//			all_types.push_back(".jpg");
//			all_types.push_back(".jpeg");
//			all_types.push_back(".bmp");
//			for (size_t i = 0; i < all_types.size(); i++)
//			{
//				std::vector<std::string> images_path_temp;
//				cv::glob(input_folder_path + "/*" + all_types[i], images_path_temp, true);
//				images_path.insert(images_path.end(), images_path_temp.begin(), images_path_temp.end());
//			}
//
//			if (images_path.empty())
//			{
//				std::cout << "No image found in the folder!" << endl;
//				return -1;
//			}
//			return 0;
//		}
//
//		/// @function: infer all images in a folder with one model
//		/// @return: 0 if success, -1 if failed
//		/// @programmer: xlxlqqq
//		/// @date: 2024.08.17
//		/// @version: 1.0
//		/// @note: do not end with "/"!!!
//		int infer_folder_images(std::string input_folder_path, std::string model_path)
//		{
//			// get all images in the folder
//			std::vector<std::string> images_path;
//			int flag_get_imagepath = get_all_images_in_folder(input_folder_path, images_path);
//			if (-1 == flag_get_imagepath)
//			{
//				return -1;
//			}
//
//			for (size_t i = 0; i < images_path.size(); i++)
//			{
//				int class_index;
//				infer_one_image(images_path[i], model_path, class_index);
//			}
//			return 0;
//		}
//
//		/// @function: test a folder with one model and calulate the correct rate
//		/// @return: 0 if success, -1 if failed
//		/// @programmer: xlxlqqq
//		/// @date: 2024.08.17
//		/// @version: 1.0
//		/// @note: images from input_folder_path should be in classified in one class type
//		/// @para<input_folder_path>: the path of the folder to be tested, do not end with "/"
//		/// @para<model_path>: the path of the model to be used
//		/// @para<correct_rate>: the correct rate of the test
//		int test_by_testdata(std::string input_folder_path, std::string model_path, float &correct_rate)
//		{
//			// get all images in the folder
//			std::vector<std::string> images_path;
//			int flag_get_imagepath = get_all_images_in_folder(input_folder_path, images_path);
//			if (-1 == flag_get_imagepath)
//			{
//				std::cout << "No image found in the folder!" << endl;
//				return -1;
//			}
//
//			// get correct class type from folder name
//			stringstream ss(input_folder_path);
//			char c = '/'; 
//			vector<string> results; 
//			string str; 
//			while (getline(ss, str, c)) 
//			{
//				results.push_back(str);
//			}
//			
//			std::string class_type_string;
//			class_type_string = results[results.size() - 1];
//			for (size_t k = 0; k < class_type_string.size(); k++)
//			{
//				if ((class_type_string[k] < '0') || (class_type_string[k] > '9'))
//				{
//					cout << "The name of the folder should be one of the class type!" << endl;
//					return -1;
//				}
//			}
//			int class_type = atoi(class_type_string.c_str());
//
//			// infer the class type and calculate correct rate
//			int correct_count = 0;
//			for (size_t i = 0; i < images_path.size(); i++)
//			{
//				int class_index;
//				infer_one_image(images_path[i], model_path, class_index);
//				if (class_index == class_type)
//				{
//					correct_count++;
//				}
//			}
//			correct_rate = correct_count / (float)images_path.size();
//			return 0;
//		}
//};

int main()
{
	openvino_classification openvino_class;
	int class_index;
	//
	openvino_class.infer_one_image("./electronisys/input_images/test.png", "./electronisys/models/model.xml", class_index);
	//cout << "The class index of the input image is: " << class_index << endl;
	//openvino_class.infer_folder_images("./electronisys/input_images", "./electronisys/models/model.xml");

	float correct_rate = 0.0;
	int flag_test = openvino_class.test_by_testdata("./electronisys/input_images/1", "./electronisys/models/model.xml", correct_rate);
	if (-1 == flag_test)
	{
		return -1;
	}
	else
	{
		cout << "Correct rate: " << correct_rate << endl;
	}
	
	return 0;
}