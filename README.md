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

## 特别说明
OpenVINO IR可以有动态张量，即 ?CWH 动态形状，但是动态形状不一定有单张推理的快，因此不采用动态形状的输入。

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