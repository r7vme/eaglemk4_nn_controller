from ros:kinetic-ros-core

# Install pip3 and pkgs for ROS python3 compatibility.
RUN apt update && apt install -y \
        python3-pip \
        python3-catkin-pkg-modules \
        python3-yaml \
        python3-rospkg-modules

# Install Python dependencies separately
# as ROS does not support python3.
ADD requirements.txt /
RUN pip3 install -r /requirements.txt

# Add source code.
ADD . /catkin_ws/src/eaglemk4_nn_controller/

# ROS workspace and dependencies.
RUN . /opt/ros/kinetic/setup.sh && \
    cd /catkin_ws && \
    catkin_make && \
    rosdep install --from-paths src --ignore-src -y

# Finally install module itself.
RUN pip3 install -e /catkin_ws/src/eaglemk4_nn_controller

COPY ./ros_entrypoint.sh /
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
