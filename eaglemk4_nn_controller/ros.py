# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
ROS node to interact with the car.
'''

import rospy
import random
import numpy as np
import threading
import time
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image


class DrivingNode:
    IMAGE_SIZE = (80, 160, 3)

    def __init__(self):
        rospy.init_node("driving_node", disable_signals=True)

        self.lock = threading.RLock()
        self.sub = rospy.Subscriber("image",
                                    Image,
                                    self.on_image)

        self.pub = rospy.Publisher("navigation",
                                   AckermannDriveStamped,
                                   queue_size=1)

    def on_image(self, img):
        d = map(ord, img.data)
        arr = np.ndarray(shape=(img.height, img.width, 3),
                         dtype=np.uint8,
                         buffer=np.array(d))[:, :, ::-1]
        if self.image_lock.acquire(True):
            self.img = arr
            self.image_lock.release()

    def get_sensor_size(self):
        return self.IMAGE_SIZE

    #
    # gym related
    #

    def is_game_over(self):
        if random.randint(1, 100) == 100:
            return True
        return False

    def observe(self):
        time.sleep(1)
        # Mock
        observation = np.zeros(self.IMAGE_SIZE, dtype=np.uint8)
        reward = 0.1
        done = self.is_game_over()
        info = None
        return observation, reward, done, info

    def reset(self):
        return

    def take_action(self, action):
        return

    def wait_ready(self):
        return
