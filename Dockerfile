# Use ROS 2 Foxy base image
FROM ros:foxy-ros-base

# Set up the ROS 2 workspace
RUN mkdir -p /home/dev_ws/src

# Set the workspace as the working directory
WORKDIR /home/dev_ws

# Clone Unity’s ROS-TCP-Endpoint
RUN git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint /home/dev_ws/src/ros_tcp_endpoint -b ROS2v0.7.0

# Create a custom ROS 2 package
RUN /bin/bash -c "source /opt/ros/foxy/setup.bash && \
    cd /home/dev_ws/src && \
    ros2 pkg create --build-type ament_python my_package"

# Build the workspace after adding packages
RUN /bin/bash -c "source /opt/ros/foxy/setup.bash && \
    cd /home/dev_ws && \
    colcon build"

# Source ROS 2 every time a new shell starts
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc

# Set the workspace as the working directory
WORKDIR /home/dev_ws
