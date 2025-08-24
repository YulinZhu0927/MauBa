import warnings

warnings.filterwarnings('ignore')

import msgpack
msgpack.Packer.DEFAULT_BUFFER_SIZE = 819200
# 增加MessagePack的最大二进制长度限制
msgpack.Unpacker.DEFAULT_MAX_BIN_LEN = 10485760  # 增加到10MB

import airsim
import cv2
import numpy as np
import time
import math
import re
import torch
import PIL.Image
import google.generativeai as genai
from google.generativeai import types
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection




# ----------------------- 配置部分 -----------------------
# 配置 Gemini API（请替换成你自己的 API Key）
genai.configure(api_key='AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88')
gemini = genai.GenerativeModel('gemini-2.0-pro-exp')

# 初始化 GroundingDINO 模型
model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id, disable_custom_kernels=True
).to(device)

# 连接 AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
camera_name = "0"
scene_image_type = airsim.ImageType.Scene

# ----------------------- 辅助函数 -----------------------
# 修复AirSim连接和图像获取
def initialize_airsim():
    """
    安全地初始化与AirSim的连接
    """
    try:
        # 尝试连接AirSim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("成功连接到AirSim")
        return client
    except Exception as e:
        print(f"连接AirSim时出错: {e}")
        print("尝试重新连接...")
        try:
            # 有时需要重新创建客户端
            time.sleep(2)
            client = airsim.MultirotorClient()
            client.confirmConnection()
            print("重新连接成功")
            return client
        except Exception as e:
            print(f"无法连接到AirSim: {e}")
            print("请确保AirSim模拟器正在运行")
            return None

def get_airsim_scene():
    """
    从AirSim获取RGB场景图像，处理可能的错误情况
    
    参数:
        client: AirSim客户端对象
    返回:
        PIL.Image: 成功时返回PIL格式图像，失败时返回None
    """
    try:
        # 使用更可靠的方式请求图像
        responses = client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
        ])
        
        if not responses or len(responses) == 0:
            print("未收到AirSim图像响应")
            return None
            
        response = responses[0]
        if response.width <= 0 or response.height <= 0:
            print("收到的图像尺寸无效")
            return None
            
        # 将图像数据转换为NumPy数组
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        
        # 检查图像数据是否为空或太小
        if len(img1d) < 100:  # 非常小的数据量肯定是无效的
            print(f"图像数据太小: {len(img1d)} 字节")
            return None

        expected_size_3 = response.height * response.width * 3
        expected_size_4 = response.height * response.width * 4

        if len(img1d) == expected_size_4:
            # 如果数据是RGBA格式，则去掉Alpha通道
            img_rgba = img1d.reshape(response.height, response.width, 4)
            img_rgb = img_rgba[:, :, :3]
        elif len(img1d) == expected_size_3:
            # 如果数据是RGB格式，则直接重塑
            img_rgb = img1d.reshape(response.height, response.width, 3)
            
        # 重塑和转换图像
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_pil = PIL.Image.fromarray(img_rgb)
        return img_pil
        
    except Exception as e:
        print(f"获取AirSim图像时出错: {e}")
        return None

def get_depth_images():
    """
    从AirSim获取深度图，处理可能的错误情况
    
    参数:
        client: AirSim客户端对象
    返回:
        tuple: (depth_planar, depth_perspective) 深度图数组
    """
    try:
        requests = [
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False),
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True, False)
        ]
        
        responses = client.simGetImages(requests)
        
        if responses is None or len(responses) < 2:
            print("未收到深度图像响应或响应不完整")
            return None, None
            
        # 处理平面深度图
        dp_response = responses[0]
        depth_planar = None
        if dp_response.width > 0 and dp_response.height > 0:
            # 验证深度数据
            if not hasattr(dp_response, 'image_data_float'):
                print("未收到平面深度图的浮点数据")
            else:
                depth_planar = np.array(dp_response.image_data_float, dtype=np.float32)
                if depth_planar.size > 0:  # 确保数组不为空
                    depth_planar = depth_planar.reshape(dp_response.height, dp_response.width)
                else:
                    print("平面深度图数据为空")
        
        # 处理透视深度图
        dper_response = responses[1]
        depth_perspective = None
        if dper_response.width > 0 and dper_response.height > 0:
            if not hasattr(dper_response, 'image_data_float'):
                print("未收到透视深度图的浮点数据")
            else:
                depth_perspective = np.array(dper_response.image_data_float, dtype=np.float32)
                if depth_perspective.size > 0:  # 确保数组不为空
                    depth_perspective = depth_perspective.reshape(dper_response.height, dper_response.width)
                else:
                    print("透视深度图数据为空")
                
        return depth_planar, depth_perspective
        
    except Exception as e:
        print(f"获取深度图像时出错: {e}")
        return None, None

def extract_python_code(content):
    """
    从 Gemini 输出中提取 Markdown 格式的 Python 代码块。
    """
    code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)
        if full_code.startswith("python"):
            full_code = full_code[7:]
        return full_code
    else:
        return None

def get_object_tags(img_pil):
    """
    使用 Gemini 模型提取图像中的物体标签，要求输出格式为：每个标签小写并以句号结尾。
    """
    response = gemini.generate_content([
        img_pil,
        (
            "Return tags for all objects as the following format in the image you can see. "
            "e.g., if you see a cat and two dogs, then response: cat. dog. "
            "Note that all the tags should be lowercased and end with a dot."
        ),
    ])
    return process_tags_to_query(response.text)

def process_tags_to_query(tags_text):
    """
    将 Gemini 返回的标签文本处理成 GroundingDINO 所需的查询字符串。
    按句号分割，并确保每个标签以句号结尾，最后以空格连接。
    """
    tags = [tag.strip() for tag in tags_text.split('.') if tag.strip()]
    processed_tags = [tag.lower() + '.' for tag in tags]
    query = ' '.join(processed_tags)
    return query

def grounding_dino_detection(img_pil, text_query):
    """
    使用 GroundingDINO 根据文本查询对图像进行目标检测，
    返回检测框（[x1, y1, x2, y2]）、置信度、标签以及标注后的图像，
    其中在标注图像上绘制了检测框、标签名称和置信度。
    """
    inputs = processor(images=img_pil, text=text_query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[img_pil.size[::-1]]  # (height, width)
    )
    detected = results[0]
    boxes = detected["boxes"].tolist()
    scores = detected["scores"].tolist()
    labels = detected["labels"]
    # 在图像上标注检测结果
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    for bbox, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(img_cv, text, (x1, max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    annotated_img = PIL.Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return boxes, scores, labels, annotated_img

def visualize_depth_map(depth_map):
    # 将深度标准化为0-255范围
    norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    norm_depth = norm_depth.astype(np.uint8)
    # 应用伪彩色映射
    depth_colormap = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
    # 显示
    cv2.imshow('Depth Map', depth_colormap)
    cv2.waitKey(1)

# 在主循环中可以调用这个函数进行场景处理和目标分析

def main():
    """
    无人机视觉模块主函数 - 集成Gemini VLM、GroundingDINO目标检测和深度分析
    """
    print("=== 无人机视觉模块启动 ===")
    
    # 初始化AirSim连接
    print("* 连接到AirSim...")
    client = initialize_airsim()
    if client is None:
        print("无法连接到AirSim，程序退出")
        return
    
    # 全局变量
    global camera_name
    camera_name = "0"  # 使用默认相机
    
    # 尝试解锁并启用无人机API
    try:
        client.enableApiControl(True)
        client.armDisarm(True)
        print("* 无人机控制已启用")
    except:
        print("* 仅视觉模式运行 (无控制功能)")
    
    print("* 加载GroundingDINO模型...")
    print("* 加载Gemini VLM...")
    print("* 系统就绪！")
    
    # 读取系统提示 (如果存在)
    sys_prompt = ""
    try:
        with open(r'../../../demo/Prompting-LLMs-for-Aerial-Navigation/system_prompts/airsim_tool_usage.txt', "r") as f:
            sys_prompt = f.read()
            print("* 已加载系统提示")
    except:
        print("* 未找到系统提示文件，使用默认设置")
    
    # 初始化会话
    is_initialized = True
    chat_history = []
    if sys_prompt:
        chat_history = [
            {
                'role': "user",
                "parts": [{
                    "text": sys_prompt
                }],
            }
        ]
    
    # 控制变量
    running = True
    detection_mode = False  # 目标检测模式
    continuous_mode = False  # 连续处理模式
    display_depth = False   # 显示深度图
    
    # 主循环
    print("\n可用命令:")
    print("!detect - 执行单次目标检测和深度分析")
    print("!auto - 切换连续检测模式")
    print("!depth - 切换深度图显示")
    print("!help - 显示帮助信息")
    print("!quit/!exit - 退出程序")
    print("其他输入将发送给Gemini进行处理\n")
    
    while running:
        # 连续模式下自动检测和分析
        if continuous_mode:
            results = process_scene_with_detection(client)
            if results:
                print(f"检测到 {len(results['depth_results'])} 个目标")
                # 显示最近的三个目标详细信息
                sorted_results = sorted(results['depth_results'], 
                                        key=lambda x: x['depth_distance'] if x['depth_distance'] is not None else float('inf'))
                for i, obj in enumerate(sorted_results[:3]):
                    if obj["depth_distance"] is not None:
                        print(f"目标 {i+1}: {obj['label']} - 距离: {obj['depth_distance']:.2f}m, 角度: {obj['yaw_angle']:.1f}°")
            time.sleep(0.5)  # 控制检测频率
        
        # 等待用户命令
        if not continuous_mode or cv2.waitKey(1) & 0xFF == ord('c'):  # 'c' 键暂停连续模式
            command = input("\n输入命令: ")
            
            # 命令处理
            if command.lower() in ["!quit", "!exit"]:
                running = False
                break
                
            elif command.lower() == "!detect":
                # 单次目标检测和深度分析
                print("执行目标检测和深度分析...")
                results = process_scene_with_detection(client)
                if results:
                    print(f"分析完成，检测到 {len(results['depth_results'])} 个目标")
                
            elif command.lower() == "!auto":
                # 切换连续检测模式
                continuous_mode = not continuous_mode
                print(f"连续检测模式: {'开启' if continuous_mode else '关闭'}")
                
            elif command.lower() == "!depth":
                # 切换深度图显示
                display_depth = not display_depth
                print(f"深度图显示: {'开启' if display_depth else '关闭'}")
                
            elif command.lower() == "!help":
                # 显示帮助信息
                print("\n可用命令:")
                print("!detect - 执行单次目标检测和深度分析")
                print("!auto - 切换连续检测模式")
                print("!depth - 切换深度图显示")
                print("!help - 显示帮助信息")
                print("!quit/!exit - 退出程序")
                print("其他输入将发送给Gemini进行处理")
                
            else:
                # 获取场景图像
                img_pil = get_airsim_scene(client)
                if img_pil is None:
                    print("未能从 AirSim 获取场景图像。")
                    continue
                
                # 首次运行时添加系统提示
                if is_initialized and sys_prompt:
                    command = sys_prompt + command
                is_initialized = False
                
                # 调用Gemini处理用户命令
                print("正在处理您的请求...")
                try:
                    response = gemini.generate_content([
                        command,
                        img_pil
                    ], stream=True)
                    response.resolve()  # 等待完整响应
                    result_text = response.text
                    print("\nGemini回应:", result_text)
                    
                    # 尝试从输出中提取Python代码块
                    code = extract_python_code(result_text)
                    if code is not None:
                        print("\n检测到代码块，是否执行? (y/n): ", end="")
                        execute = input().lower()
                        if execute == 'y':
                            print("\n执行代码:")
                            try:
                                # 创建本地变量来帮助执行代码
                                local_vars = {
                                    'client': client,
                                    'gemini': gemini,
                                    'camera_name': camera_name,
                                    'get_airsim_scene': lambda: get_airsim_scene(client),
                                    'get_depth_images': lambda: get_depth_images(client)
                                }
                                # 执行代码
                                exec(code, globals(), local_vars)
                                print("代码执行完成")
                            except Exception as e:
                                print(f"代码执行错误: {e}")
                except Exception as e:
                    print(f"Gemini API 错误: {e}")
        
        # 显示深度图（如果启用）
        if display_depth:
            depth_planar, depth_perspective = get_depth_images(client)
            if depth_planar is not None:
                # 可视化深度图
                visualize_depth_map(depth_planar)
    
    # 清理并退出
    print("正在关闭视觉模块...")
    if display_depth:
        cv2.destroyAllWindows()
    try:
        client.armDisarm(False)
        client.enableApiControl(False)
    except:
        pass
    print("视觉模块已关闭")

def process_scene_with_detection():
    """
    处理当前场景，进行目标检测和深度分析
    
    参数:
        client: AirSim客户端对象
    """
    # 获取场景图像
    img_pil = get_airsim_scene(client)
    if img_pil is None:
        print("未能从 AirSim 获取场景图像。")
        return None
    
    # 使用Gemini获取物体标签
    text_query = get_object_tags(img_pil)
    print(f"检测到的物体标签: {text_query}")
    
    # 使用GroundingDINO进行目标检测
    boxes, scores, labels, annotated_img = grounding_dino_detection(img_pil, text_query)
    
    # 如果检测到目标，计算深度和偏航角
    if boxes:
        # 传递图像尺寸作为scene_shape
        depth_results = estimate_depth_yaw(boxes, labels, img_pil.size, client)
        
        # 在标注图像上添加深度和偏航角信息
        img_cv = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)
        for result in depth_results:
            if result["depth_distance"] is not None and result["yaw_angle"] is not None:
                x1, y1, x2, y2 = result["bbox"]
                depth_text = f"D: {result['depth_distance']:.2f}m"
                yaw_text = f"Y: {result['yaw_angle']:.1f}°"
                
                # 在检测框下方添加深度和偏航角信息
                cv2.putText(img_cv, depth_text, (int(x1), int(y2+20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                cv2.putText(img_cv, yaw_text, (int(x1), int(y2+40)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        
        final_img = PIL.Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # 显示标注后的图像
        cv2.imshow('Detection Results', cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        
        # 返回检测与深度分析结果
        return {
            "detected_objects": len(boxes),
            "labels": labels,
            "depth_results": depth_results,
            "annotated_image": final_img
        }
    else:
        print("未检测到任何目标")
        return None

def estimate_depth_yaw(boxes, labels, scene_shape, client):
    """
    对每个检测到的目标，根据深度图计算目标的深度和偏航角。
    注意：boxes 的坐标基于场景图（例如1920×1080），而深度图分辨率较低（例如640×360）。
    因此需要按比例转换坐标。
    
    参数:
        boxes: 检测框列表 [x1, y1, x2, y2]
        labels: 标签列表
        scene_shape: 场景图像尺寸 (width, height)
        client: AirSim客户端对象
    返回:
        包含bbox、深度值、偏航角、场景坐标和深度图坐标的字典列表
    """
    depth_planar, depth_perspective = get_depth_images(client)
    if depth_planar is None or depth_perspective is None:
        print("无法获取深度图。")
        return [{"bbox": bbox, "label": label, "depth_distance": None, "camera_distance": None, 
                "yaw_angle": None, "center_scene": None, "center_depth": None} 
                for bbox, label in zip(boxes, labels)]

    scene_h, scene_w = scene_shape[1], scene_shape[0]  # PIL Image size is (width, height)
    depth_h, depth_w = depth_planar.shape[0], depth_planar.shape[1]
    
    # 打印调试信息
    print(f"Scene dimensions: {scene_w}x{scene_h}")
    print(f"Depth dimensions: {depth_w}x{depth_h}")
    
    scale_x, scale_y = depth_w / scene_w, depth_h / scene_h
    print(f"Scale factors: x={scale_x}, y={scale_y}")

    results = []
    for i, (bbox, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, bbox)
        center_x_scene = int((x1 + x2) / 2)
        center_y_scene = int((y1 + y2) / 2)
        
        # 转换为深度图坐标，确保在边界内
        center_x_depth = min(max(0, int(center_x_scene * scale_x)), depth_w - 1)
        center_y_depth = min(max(0, int(center_y_scene * scale_y)), depth_h - 1)
        
        # 不使用单个像素点，而是使用目标中心区域的平均深度
        box_width = max(1, int((x2 - x1) * scale_x * 0.2))  # 使用检测框宽度的20%
        box_height = max(1, int((y2 - y1) * scale_y * 0.2))  # 使用检测框高度的20%
        
        # 确保采样区域在深度图范围内
        x_min = max(0, center_x_depth - box_width)
        x_max = min(depth_w - 1, center_x_depth + box_width)
        y_min = max(0, center_y_depth - box_height)
        y_max = min(depth_h - 1, center_y_depth + box_height)
        
        # 计算目标区域的平均深度
        depth_roi = depth_planar[y_min:y_max+1, x_min:x_max+1]
        camera_roi = depth_perspective[y_min:y_max+1, x_min:x_max+1]
        
        # 过滤掉异常值（例如65500+）
        valid_depths = depth_roi[depth_roi < 65000]
        valid_cameras = camera_roi[camera_roi < 65000]
        
        if len(valid_depths) > 0 and len(valid_cameras) > 0:
            depth_distance = np.median(valid_depths)  # 使用中值更健壮
            camera_distance = np.median(valid_cameras)
            
            # 调试输出
            print(f"Object {i} ({label}):")
            print(f"  Scene center: ({center_x_scene}, {center_y_scene})")
            print(f"  Depth center: ({center_x_depth}, {center_y_depth})")
            print(f"  Sample region: x=[{x_min}:{x_max}], y=[{y_min}:{y_max}]")
            print(f"  Valid depth samples: {len(valid_depths)}")
            print(f"  Depth range: {np.min(valid_depths):.2f} - {np.max(valid_depths):.2f}")
            print(f"  Median Depth: {depth_distance:.2f}, Camera: {camera_distance:.2f}")
            
            # 计算偏航角
            # 基于图像坐标与中心的水平偏移（更直观的方法）
            horizontal_offset = center_x_scene - (scene_w / 2)
            # 将偏移转换为角度（可以用视场角FOV来计算更准确的值）
            FOV = 90  # 假设水平视场角为90度，需要根据实际摄像头参数调整
            pixel_to_degree = FOV / scene_w
            yaw_angle = horizontal_offset * pixel_to_degree
            
            print(f"  Horizontal offset: {horizontal_offset:.1f} pixels")
            print(f"  Yaw angle: {yaw_angle:.2f}°")
        else:
            depth_distance = None
            camera_distance = None
            yaw_angle = None
            print(f"Object {i} ({label}): No valid depth measurements")
            print(f"  ROI shape: {depth_roi.shape}")
            if depth_roi.size > 0:
                print(f"  Depth values range: {np.min(depth_roi):.2f} - {np.max(depth_roi):.2f}")

        results.append({
            "bbox": bbox,
            "label": label,
            "depth_distance": depth_distance,
            "camera_distance": camera_distance,
            "yaw_angle": yaw_angle,
            "center_scene": (center_x_scene, center_y_scene),
            "center_depth": (center_x_depth, center_y_depth)
        })
    return results

tool_config = [{
    "name": "process_scene_with_detection",
    "description": "Detect objects and measure their distances and yaw angles in the current AirSim drone scene.",
    "parameters": {
        "type": "object",
        "properties": {
            "no_parameters": {
                "type": "string",
                "description": "This function requires no parameters. Always pass 'none'."
            }
        },
        "required": ["no_parameters"]
    }
}]

# 如果是主程序，则运行main函数
# if __name__ == "__main__":
#     # main()
#     prompt = "Please analyze the drone scene and return detected objects with their distances and yaw angles."
#     response = gemini.generate_content([prompt], tools={"function_declarations": tool_config})
#
#     # 检查Gemini响应，调用实际函数
#     if response.candidates[0].content.parts[0].function_call.name == "process_scene_with_detection":
#         scene_analysis_results = process_scene_with_detection()
#         print(scene_analysis_results)