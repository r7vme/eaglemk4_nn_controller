# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
ROS node to interact with the car.
'''

import rospy
import numpy as np
import threading
import time
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image, Joy

# TODO: output driving values asynchronously.
class DrivingNode:
    '''
    DrivingNode node implements ROS node to drive a car.

    This node managed via OpenAI gym interface.
    '''

    IMAGE_SIZE = (80, 160, 3)
    IMAGE_MSG_ENCODING = "rgb8"
    HZ = 20
    TASK_TEST = 0
    TASK_TRAIN = 1

    def __init__(self):
        rospy.init_node("driving_node", disable_signals=True)
        # Max angle in radians. Ackermann uses sttering angle in radians.
        self.angle_scale = 0.3
        # Joy's "triangle" switches task between "train" and "test".
        self.button_task = 0
        # Joy's "right bumper" enables autopilot. NOTE: Unpressing means
        # human took control, which finishes episode.
        self.button_autopilot = 5
        # Use static throttle.
        self.static_throttle = 0.25

        self.image_lock = threading.RLock()

        # Subscribe to camera images.
        self.image_sub = rospy.Subscriber("image",
                                          Image,
                                          self.on_image)

        # Subscribe to joystick.
        self.image_sub = rospy.Subscriber("joy",
                                          Joy,
                                          self.on_joy)

        # Publish steering and throttle.
        self.nav_pub = rospy.Publisher("navigation",
                                       AckermannDriveStamped,
                                       queue_size=1)

        # Init runtime variables.
        self.autopilot = False
        self.image_array = np.zeros(self.IMAGE_SIZE)
        self.last_obs = None
        self.last_throttle = 0.0
        self.task = self.TASK_TEST
        self.task_toggle_time = time.time()

    # on_image stores image from camera in self.image_array
    def on_image(self, msg):
        assert msg.encoding == self.IMAGE_MSG_ENCODING
        assert msg.height == self.IMAGE_SIZE[0]
        assert msg.width == self.IMAGE_SIZE[1]

        channels = self.IMAGE_SIZE[2]
        dtype = np.dtype(np.uint8)
        shape = self.IMAGE_SIZE

        data = np.fromstring(msg.data, dtype=np.uint8).reshape(shape)
        data.strides = (
         msg.step,
         dtype.itemsize * channels,
         dtype.itemsize
        )

        assert data.shape == self.IMAGE_SIZE
        if self.image_lock.acquire(True):
            self.image_array = data
            self.image_lock.release()

    # on_joy handles buttons.
    def on_joy(self, msg):
        # Handle task button.
        if (time.time() - self.task_toggle_time) > 1 and \
                msg.buttons[self.button_task]:
            self.task = abs(self.task - 1)
            print("Task: ", "test" if self.task == self.TASK_TEST else "train")
            self.task_toggle_time = time.time()

        # Handle autopilot button.
        if msg.buttons[self.button_autopilot] and not self.autopilot:
            self.autopilot = True
            print("Autopilot: enabled")
        elif not msg.buttons[self.button_autopilot] and self.autopilot:
            self.autopilot = False
            print("Autopilot: disabled")

    def get_sensor_size(self):
        return self.IMAGE_SIZE

    def _rel_to_rad(self, angle):
        return angle * self.angle_scale

    #
    # gym related
    #

    def is_game_over(self):
        return not self.autopilot

    def observe(self):
        self.last_obs = self.image_array
        observation = self.image_array
        done = self.is_game_over()
        reward = self._calc_reward(done)
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.image_array = np.zeros(self.IMAGE_SIZE)
        self.last_obs = None
        self.last_throttle = 0.0
        self._wait_reset()

    def take_action(self, action):
        throttle = self.static_throttle
        steering_angle = self._rel_to_rad(action[0])

        # Prepare message.
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = throttle
        msg.drive.steering_angle = steering_angle

        # Update last_throttle.
        self.last_throttle = throttle

        self.nav_pub.publish(msg)

    # _wait_reset will wait until human will enable autopilot mode.
    def _wait_reset(self):
        print("waiting reset")
        while not self.autopilot:
            time.sleep(1.0 / self.HZ)

    # _calc_reward returns reward equal to car velocity
    # or zero if episode has finished.
    def _calc_reward(self, done):
        if done:
            return 0.0
        return self.last_throttle * (1.0 / self.HZ)
