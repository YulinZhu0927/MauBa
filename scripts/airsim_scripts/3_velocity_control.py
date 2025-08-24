import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()

client.moveToZAsync(-3, 1).join()
client.moveToPositionAsync(5, 0, -3, 2).join()
client.moveToPositionAsync(5, 5, -3, 4).join()
client.moveToPositionAsync(0, 5, -3, 3).join()
client.moveToPositionAsync(0, 0, -3, 1).join()

client.hoverAsync().join()
time.sleep(2)

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)