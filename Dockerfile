FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Install Gazebo Classic, Nav2, ros2_control, DLIO deps, and Python/Tk GUI packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-xacro \
    ros-humble-pcl-ros \
    ros-humble-pcl-conversions \
    libpcl-dev \
    libeigen3-dev \
    python3-numpy \
    python3-yaml \
    python3-tk \
    python3-pil \
    python3-pil.imagetk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws
COPY src/ /ros2_ws/src/

# Build DLIO and a1_description packages
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && ros2 launch a1_description navigation.launch.py"]
