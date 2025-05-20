# Use ROS 2 Foxy base image
FROM ros:foxy-ros-base

# Install dependencies
RUN apt update && apt install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-foxy-cv-bridge \
    git \
    build-essential \
    cmake \
    python3-dev


# Upgrade pip and install Python packages
RUN python3 -m pip install --upgrade pip && \
    pip3 install \
    opencv-python \
    numpy \
    pandas

# Set up the ROS 2 workspace
RUN mkdir -p /home/dev_ws/src
WORKDIR /home/dev_ws

# Clone Unity’s ROS-TCP-Endpoint
RUN git clone -b ROS2v0.7.0 https://github.com/Unity-Technologies/ROS-TCP-Endpoint /home/dev_ws/src/ros_tcp_endpoint

# Build the workspace
RUN /bin/bash -c "source /opt/ros/foxy/setup.bash && \
    cd /home/dev_ws && \
    colcon build"

# Source ROS 2 setup files in every shell
RUN echo "source /opt/ros/foxy/setup.bash" >> /root/.bashrc && \
    echo "source /home/dev_ws/install/setup.bash" >> /root/.bashrc

# Set the workspace as the working directory (optional, just good for dev)
WORKDIR /home/dev_ws
