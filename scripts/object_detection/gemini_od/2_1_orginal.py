import warnings

warnings.filterwarnings('ignore')

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
def get_airsim_scene():
    """
    从 AirSim 获取 RGB 场景图像（Scene），返回 PIL.Image 格式图像。
    """
    raw_scene = client.simGetImage(camera_name, scene_image_type)
    if raw_scene is None:
        return None
    np_img = np.frombuffer(raw_scene, dtype=np.uint8)
    img_cv = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img_cv is None:
        return None
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = PIL.Image.fromarray(img_rgb)
    return img_pil

def get_depth_images():
    """
    从 AirSim 获取深度图：DepthPlanar 和 DepthPerspective，
    使用浮点格式数据（pixels_as_float=True, compress=False）。
    返回：
        depth_planar: 2D 浮点数组（例如分辨率 640×360）
        depth_perspective: 2D 浮点数组（例如分辨率 640×360）
    """
    requests = [
        airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False),
        airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True, False)
    ]
    responses = client.simGetImages(requests)
    if responses is None or len(responses) < 2:
        return None, None
    dp_response = responses[0]
    if dp_response.width > 0 and dp_response.height > 0:
        depth_planar = np.array(dp_response.image_data_float, dtype=np.float32)
        depth_planar = depth_planar.reshape(dp_response.height, dp_response.width)
    else:
        depth_planar = None
    dper_response = responses[1]
    if dper_response.width > 0 and dper_response.height > 0:
        depth_perspective = np.array(dper_response.image_data_float, dtype=np.float32)
        depth_perspective = depth_perspective.reshape(dper_response.height, dper_response.width)
    else:
        depth_perspective = None
    return depth_planar, depth_perspective

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

def estimate_depth_yaw(boxes, depth_planar, depth_perspective, scene_shape):
    """
    对每个检测到的目标，根据深度图计算目标的深度和偏航角。
    注意：boxes 的坐标基于场景图（例如1920×1080），而深度图分辨率较低（例如640×360）。
    因此需要按比例转换坐标。
    返回包含 bbox、深度值、偏航角、场景坐标和深度图坐标的字典列表。
    """
    scene_h, scene_w = scene_shape[0], scene_shape[1]
    # 获取深度图尺寸（假设 depth_planar 和 depth_perspective 分辨率一致）
    if depth_planar is not None:
        depth_h, depth_w = depth_planar.shape[0], depth_planar.shape[1]
    else:
        depth_h, depth_w = 0, 0
    scale_x = depth_w / scene_w
    scale_y = depth_h / scene_h
    results = []
    for bbox in boxes:
        x1, y1, x2, y2 = map(int, bbox)
        center_x_scene = int((x1 + x2) / 2)
        center_y_scene = int((y1 + y2) / 2)
        # 转换为深度图坐标
        center_x_depth = int(center_x_scene * scale_x)
        center_y_depth = int(center_y_scene * scale_y)
        center_x_depth = max(0, min(center_x_depth, depth_w - 1))
        center_y_depth = max(0, min(center_y_depth, depth_h - 1))
        depth_distance = depth_planar[center_y_depth, center_x_depth] if depth_planar is not None else None
        camera_distance = depth_perspective[center_y_depth, center_x_depth] if depth_perspective is not None else None
        yaw_angle = None
        if depth_distance is not None and camera_distance is not None and camera_distance != 0:
            ratio = depth_distance / camera_distance
            ratio = max(min(ratio, 1.0), -1.0)
            angle_rad = math.acos(ratio)
            yaw_angle = math.degrees(angle_rad)
            if center_x_scene < scene_w / 2:
                yaw_angle = -yaw_angle
        results.append({
            "bbox": bbox,
            "depth_distance": depth_distance,
            "camera_distance": camera_distance,
            "yaw_angle": yaw_angle,
            "center_scene": (center_x_scene, center_y_scene),
            "center_depth": (center_x_depth, center_y_depth)
        })
    return results


# ----------------------- 主循环 -----------------------
while True:
    command = input("请输入命令（或输入 !quit / !exit 退出）：")
    if command.lower() in ["!quit", "!exit"]:
        break

    # 获取场景图像
    img_pil = get_airsim_scene()
    if img_pil is None:
        print("未能从 AirSim 获取场景图像。")
        time.sleep(0.1)
        continue

    # 获取深度图（独立获取）
    depth_planar, depth_perspective = get_depth_images()
    if depth_planar is None or depth_perspective is None:
        print("未能从 AirSim 获取深度图。")
        time.sleep(0.1)
        continue

    # 使用 Gemini 提取图像中的物体标签
    gemini_tags = get_object_tags(img_pil)
    print("Gemini 输出的标签：", gemini_tags)

    # 将标签处理成 GroundingDINO 的查询字符串
    text_query = process_tags_to_query(gemini_tags)
    print("GroundingDINO 检测查询：", text_query)

    # 调用 GroundingDINO 进行目标检测，获取检测框、置信度、标签和标注后的图像
    boxes, scores, labels, annotated_img = grounding_dino_detection(img_pil, text_query)
    print("检测到的框：", boxes)

    # 估计每个检测目标的深度和偏航角（坐标转换后计算）
    scene_np = np.array(img_pil)
    detection_info = estimate_depth_yaw(boxes, depth_planar, depth_perspective, scene_np.shape)
    for info in detection_info:
        print("目标信息：", info)

    # 显示标注后的图像
    annotated_img.show()

    # 如果 Gemini 输出中包含可执行的 Python 代码，可以尝试提取并执行
    code = extract_python_code(gemini_tags)
    if code is not None:
        print("提取到的代码：\n", code)
        try:
            exec(code)
        except Exception as e:
            print("执行代码时发生错误：", e)

    time.sleep(0.5)
