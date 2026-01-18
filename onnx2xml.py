import subprocess

def convert_onnx_to_openvino(onnx_model_path, output_dir):
    # 构建 Model Optimizer 命令
    mo_command = [
        "mo",
        "--input_model", onnx_model_path,
        "--output_dir", output_dir,
        "--framework", "onnx",
        "--input", "images[16,3,640,640]"  # 支持动态 batch
    ]
    
    # 运行 Model Optimizer 工具
    subprocess.run(mo_command, check=True)

# 示例使用
onnx_model_path = "./weights/yolo_paints/best.onnx"  # 替换为你的 ONNX 模型路径
output_dir = "./weights/yolo_paints/"              # 输出文件夹路径
convert_onnx_to_openvino(onnx_model_path, output_dir)

