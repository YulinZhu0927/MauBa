import airsim
import cv2
import numpy as np
import time
import google.generativeai as genai
import PIL.Image
import re

# 配置 Gemini API（请确保 API Key 有效）
genai.configure(api_key='AIzaSyDNXUAZMUh6KYONW59rYHOqIi5tvQ1Pn88')
gemini = genai.GenerativeModel('gemini-2.0-pro-exp')

# 连接到 AirSim 模拟器
client = airsim.MultirotorClient()
client.confirmConnection()

# 设置摄像头名称
camera_name = "0"

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
    # command = input("Enter your command (or !quit to exit): ")
    # if command.lower() in ["!quit", "!exit"]:
    #     break

    # 请求场景图像和深度图（使用 simGetImages，pixels_as_float=True 返回浮点数数据）
    images_requests = [
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False),
        airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True, False)
    ]
    images = client.simGetImages([

    ])

    if images is None or len(images) < 3:
        print("No image received from AirSim.")
        time.sleep(0.1)
        continue

    # 处理场景图像
    scene_response = images[0]
    scene_np = np.frombuffer(scene_response.image_data_uint8, dtype=np.uint8)
    scene_img = cv2.imdecode(scene_np, cv2.IMREAD_COLOR)
    if scene_img is None:
        print("Failed to decode scene image.")
        continue
    # 转换为 RGB 用于 PIL 显示
    scene_img_rgb = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
    img_pil = PIL.Image.fromarray(scene_img_rgb)
    print("Received Scene image shape:", scene_img.shape)

    # 处理 DepthPlanar，注意 pixels_as_float=True 返回的是浮点数列表
    dp_response = images[1]
    if dp_response.width > 0 and dp_response.height > 0:
        depth_planar = np.array(dp_response.image_data_float, dtype=np.float32)
        depth_planar = depth_planar.reshape(dp_response.height, dp_response.width)
        print("DepthPlanar shape:", depth_planar.shape)
    else:
        depth_planar = None
        print("DepthPlanar not received.")

    # 处理 DepthPerspective
    dper_response = images[2]
    if dper_response.width > 0 and dper_response.height > 0:
        depth_perspective = np.array(dper_response.image_data_float, dtype=np.float32)
        depth_perspective = depth_perspective.reshape(dper_response.height, dper_response.width)
        print("DepthPerspective shape:", depth_perspective.shape)
    else:
        depth_perspective = None
        print("DepthPerspective not received.")

    # 指定显示位置的坐标（注意 numpy 数组索引为 [y, x]）
    x_coord, y_coord = 1625, 539
    if depth_planar is not None:
        if y_coord < depth_planar.shape[0] and x_coord < depth_planar.shape[1]:
            print("DepthPlanar value at (x=%d, y=%d):" % (x_coord, y_coord), depth_planar[y_coord, x_coord])
        else:
            print("Specified coordinate out of range in DepthPlanar.")
    if depth_perspective is not None:
        if y_coord < depth_perspective.shape[0] and x_coord < depth_perspective.shape[1]:
            print("DepthPerspective value at (x=%d, y=%d):" % (x_coord, y_coord), depth_perspective[y_coord, x_coord])
        else:
            print("Specified coordinate out of range in DepthPerspective.")

    # 显示原始场景图像
    cv2.namedWindow("Scene Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Scene Image", scene_img)

    # 为了显示深度图，归一化到 0-255 范围后转换为 uint8 类型
    if depth_planar is not None:
        dp_norm = cv2.normalize(depth_planar, None, 0, 255, cv2.NORM_MINMAX)
        dp_norm = np.uint8(dp_norm)
        cv2.namedWindow("DepthPlanar", cv2.WINDOW_NORMAL)
        cv2.imshow("DepthPlanar", dp_norm)
    if depth_perspective is not None:
        dper_norm = cv2.normalize(depth_perspective, None, 0, 255, cv2.NORM_MINMAX)
        dper_norm = np.uint8(dper_norm)
        cv2.namedWindow("DepthPerspective", cv2.WINDOW_NORMAL)
        cv2.imshow("DepthPerspective", dper_norm)

    print("Press any key in any image window to continue...")
    key = cv2.waitKey(0)  # 暂停等待按键
    cv2.destroyAllWindows()

    # 下面 Gemini 部分暂时注释，如有需要可恢复
    # response = gemini.generate_content([command, img_pil], stream=True)
    # response.resolve()
    # result_text = response.text
    # print("Gemini output:", result_text)
    #
    # code = extract_python_code(result_text)
    # if code is not None:
    #     print("Extracted code:\n", code)
    #     try:
    #         exec(code)
    #     except Exception as e:
    #         print("Error executing code:", e)
