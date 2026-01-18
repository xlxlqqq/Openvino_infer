本项目包含多个使用openvino推理的demo源代码

# OpenVINO for YOLOv5
## 代码文件
YOLO5.cpp

## 介绍
模型可以是静态 1 x 3 x H x W，需要将pytorch模型转化成onnx再转化成OpenVINO IR格式（xml和bin格式）

YOLOv5 最新的代码仓库已经自带了export.py可以直接从pt转化成OpenVINO IR格式


# OpenVINO for YOLOv5 batch
## 代码文件
yolo5_batch.cpp

## 介绍
使用OpenVINO一次推理多个张量，需要配合多个张量的OpenVINO IR格式的模型（模型xml文件里面输入层有写输入张量的维度）

## 模型导出
使用YOLOv5 v7.0版本的仓库，在导出时可以选择如下命令：
python export.py --dynamic --batch-size 16生成不同BatchSize的模型
如果借助onnx模型作为中间模型表示，则在onnx转化OpenVINO IR格式的时候，需要使用如下命令：
mo_command = [
    "mo",
    "--input_model", onnx_model_path,
    "--output_dir", output_dir,
    "--framework", "onnx",
    "--input", "images[16,3,640,640]"  # 支持动态 batch
]


## 特别说明
OpenVINO IR可以有动态张量，即 ?CWH 动态形状，但是动态形状不一定有单张推理的快，因此不采用动态形状的输入。

## 可能的优化空间
### OpenVINO 提供了 前处理方法，能够合理调用 SIMD 进行优化和cache优化，可以进行尝试
inferBatch::inferBatch(...) {

    ov::Model model = core_.read_model(model_path_);

    ov::preprocess::PrePostProcessor ppp(model);

    ppp.input()
       .tensor()
       .set_element_type(ov::element::u8)
       .set_layout("NHWC")
       .set_color_format(ov::preprocess::ColorFormat::BGR);

    ppp.input()
       .preprocess()
       .convert_color(ov::preprocess::ColorFormat::RGB)
       .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
       .convert_element_type(ov::element::f32)
       .scale(255.0f);

    ppp.input()
       .model()
       .set_layout("NCHW");

    model = ppp.build();

    compiled_model_ = core_.compile_model(model, "CPU");
    infer_request_ = compiled_model_.create_infer_request();
}

### 工程健壮性
外部接口均由inferBatch.infer()执行，不够健壮。建议单独设计前处理，后处理，infer推理类。

# OpenVINO for PTV2
## 代码文件
PTV2.cpp

## 介绍
使用OpenVINO推理PTV2模型和GRP-PTv2模型

# OpenVINO for pointnet
## 代码文件
pointnet++.cpp

## 介绍
使用OpenVINO推理pointnet模型

# OpenVINO for pointnet
## 代码文件
electronisys.cpp

## 介绍
自己设计的模型，用来检测电泳件
