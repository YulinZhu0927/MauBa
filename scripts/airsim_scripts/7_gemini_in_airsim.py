import airsim
import cv2
import numpy as np
import time
import google.generativeai as genai
import PIL.Image
import re

# print(0)



# 配置 Gemini API（请确保 API Key 有效）
genai.configure(api_key='AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88')

# import pprint
# for model in genai.list_models():
#     pprint.pprint(model)

gemini = genai.GenerativeModel('gemini-2.0-pro-exp')

# print(1)

# 连接到 AirSim 模拟器
client = airsim.MultirotorClient()
client.confirmConnection()

# client.simGetDetections()

# print(2)

# 设置摄像头和图像类型（例如 "0" 表示默认摄像头，ImageType.Scene 表示场景图像）
camera_name = "0"
image_type = airsim.ImageType.Scene

def extract_python_code(content):
    """
    从字符串中提取 Markdown 格式的 Python 代码块。
    如果代码块以 "python" 开头，则去除该标识。
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

while True:
    command = input("Enter your command (or !quit to exit): ")
    if command.lower() in ["!quit", "!exit"]:
        break

    # 获取摄像头图像（返回压缩后的图像字节流）
    raw_images = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, True),
        # airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False)
    ])


    print(len(raw_images))

    raw_image = raw_images[0]

    raw_bps = client.simGetImages([
        # airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, True),
        airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False)
    ])
    raw_dp = raw_bps[0]


    if raw_image is None:
        print("No image received from AirSim.")
        time.sleep(0.1)
        continue

    # 将字节流转换为 numpy 数组，并解码为 OpenCV 格式图像（BGR）
    np_img = np.frombuffer(raw_image.image_data_uint8, dtype=np.uint8)

    if raw_dp.width > 0 and raw_dp.height > 0:
        depth_planar = np.array(raw_dp.image_data_float, dtype=np.float32)
        depth_planar = depth_planar.reshape(raw_dp.height, raw_dp.width)
    else:
        depth_planar = None

    if depth_planar is not None:
        print("DepthPlanar: min =", np.min(depth_planar),
              ", max =", np.max(depth_planar),
              ", mean =", np.mean(depth_planar))
    # print(np_img[539, 1625])

    img_cv = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img_cv is None:
        print("Failed to decode image.")
        continue

    # 转换 BGR 为 RGB，再转换为 PIL.Image 对象
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = PIL.Image.fromarray(img_rgb)

    # 调用 Gemini‑1.5‑pro 模型，使用用户输入的命令和当前图像作为输入（stream 模式）
    response = gemini.generate_content([
        command,
        img_pil
    ], stream=True)
    response.resolve()  # 等待完整响应
    result_text = response.text
    print("Gemini output:", result_text)

    # 尝试从输出中提取 Python 代码块
    code = extract_python_code(result_text)
    if code is not None:
        print("Extracted code:\n")
        print(code)
        try:
            exec(code)
        except Exception as e:
            print("Error executing code:", e)

# 如果有需要，也可以显示图像调试，现已注释掉
# cv2.destroyAllWindows()
