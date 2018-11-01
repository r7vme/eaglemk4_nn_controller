# eaglemk4_nn_controller

Reinforsment learning (VAE + DDPG) controller for EagleMK4 car.

[Demo](https://www.youtube.com/watch?v=6JUjDw9tfD4).

[Donkey simulation](https://github.com/r7vme/learning-to-drive-in-a-day).

# Usage

This controller expects [EagleMK4](https://github.com/r7vme/eagleMK4) or similar [MIT-racecar](https://github.com/mit-racecar/racecar) like ackermann steering robot.

In general, controller can be adapted to any ROS-based robot with monocular camera.

Inputs:
- `sensor_msgs/Image` from topic `image`.
- `sensor_msgs/Joy` from topic `joy`.

Outputs:
- `ackermann_msgs/AckermannDriveStamped` from topic `navigation`.

Joystick buttons:
- Triangle - switch tasks (train and test).
- Right bumper - when pressed enables autonomous mode. Unpressing stops the autonomous mode and stops the episode.
