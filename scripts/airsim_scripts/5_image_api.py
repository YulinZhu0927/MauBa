import airsim
import numpy as np
import cv2

client = airsim.MultirotorClient()
client.confirmConnection()

# response = client.simGetImages(camera_name, image_type, vehicle_name) # 一次获取一张

# 摄像头编号
# 0-前  1-后  2-底部  3-顶部  4-左  5-右

# 能获取到的图像类型
# 1-scene 场景（俯视）
# 2-depth planar 平面深度图
# 3-depth perspective 透视深度图
# 4-depth vis 深度可视化图，可转为RGB
# 5-disparity normalized 视差归一化图
# 6-segmentation 分割
# 7-surface normals 表面法线图
# 8-infrared 红外线图

vehicle_name = 'Drone'
# 获取RGB PNG
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Infrared, False, False),
    airsim.ImageRequest("0", airsim.ImageType.SurfaceNormals, False, False)
])

# 检查是否获得了有效响应
if responses and len(responses) > 0:
    response = responses[0]

    # 转换为numpy数组
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

    if img1d.size != 0:
        # reshape为图像格式
        img_rgb = img1d.reshape(response.height, response.width, 3)

        # 显示图像
        cv2.imshow("AirSim Image", img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("图像数据为空！")
else:
    print("未收到有效图像响应！")
