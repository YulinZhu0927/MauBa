import sys
import pygame
import time
import airsim

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Airsim controlled by pygame')
screen.fill((0, 0, 0))

vehicle_name = "Drone"
airsim_client = airsim.MultirotorClient()
airsim_client.confirmConnection()
airsim_client.enableApiControl(True, vehicle_name=vehicle_name)
airsim_client.armDisarm(True, vehicle_name=vehicle_name)
airsim_client.takeoffAsync(vehicle_name=vehicle_name).join()

vehicle_velocity = 2.0
speedup_ratio = 10.0
speedup_flag = False

vehicle_yaw_rate = 5.0

while True:
    yaw_rate = 0.0
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0

    time.sleep(0.02)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    scan_wrapper = pygame.key.get_pressed()

    if scan_wrapper[pygame.K_SPACE]: # 空格加速10倍
        scale_ratio = speedup_ratio
    else:
        scale_ratio = speedup_ratio / speedup_ratio

    if scan_wrapper[pygame.K_a] or scan_wrapper[pygame.K_d]: # A D 设置偏航速率
        yaw_rate = (scan_wrapper[pygame.K_d] - scan_wrapper[pygame.K_a]) * scale_ratio * vehicle_yaw_rate

    if scan_wrapper[pygame.K_UP] or scan_wrapper[pygame.K_DOWN]: # 上 下 设置pitch轴速度变量 x为机头方向 (NED坐标系)
        v_x = (scan_wrapper[pygame.K_UP] - scan_wrapper[pygame.K_DOWN]) * scale_ratio

    if scan_wrapper[pygame.K_LEFT] or scan_wrapper[pygame.K_RIGHT]: # 左 右 设置roll轴速度变量 y为右
        v_y = -(scan_wrapper[pygame.K_LEFT] - scan_wrapper[pygame.K_RIGHT]) * scale_ratio

    if scan_wrapper[pygame.K_w] or scan_wrapper[pygame.K_s]: # W S 设置Z轴方向 上为负
        v_z = -(scan_wrapper[pygame.K_w] - scan_wrapper[pygame.K_s]) * scale_ratio

    airsim_client.moveByVelocityBodyFrameAsync(vx=v_x, vy=v_y, vz=v_z, duration=0.02,
                                               yaw_mode=airsim.YawMode(True, yaw_or_rate=yaw_rate), vehicle_name=vehicle_name)

    if scan_wrapper[pygame.K_ESCAPE]: # ESC退出
        pygame.quit()
        sys.exit()
