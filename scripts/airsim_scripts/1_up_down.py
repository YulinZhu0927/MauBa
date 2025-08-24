import airsim

# 连接到airsim
client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True) # 允许API控制
client.armDisarm(True) # 解锁

client.takeoffAsync().join() # 起飞，.join()表示任务执行完毕后再执行下一任务
client.landAsync().join() # 降落

client.armDisarm(False) # 锁定
client.enableApiControl(False) # 解除控制