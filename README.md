# Balloon Position Detector

This project simulates a ROS2-based vision system that detects and tracks a balloon in a Unity environment using a trained PyTorch CNN model.

## 🚀 Features

- Real-time object detection using a custom-trained CNN
- ROS2 + Unity integration via ROS TCP Connector
- Outputs object position data over ROS topics for downstream control
- Dockerized environment for reproducibility and portability

## 🧠 Tech Stack

- **Python** (PyTorch, OpenCV)
- **Unity** (with ROS TCP Connector)
- **ROS2 Foxy** (inside Docker)
- **rospy / roslibpy** bridge for lightweight message passing

## 🗂️ Project Structure

```
.
├── pytorchopencv/       # CNN training & inference code
│   ├── BalloonNetCNNBox.py
│   ├── balloon_pos.py
│   ├── iou.py
│   ├── BalloonBoxDataset.py
│   └── balloon_pos.pth  # (ignored, too large for GitHub)
├── nodes/               # ROS Python nodes
│     ├── BalloonNetBox.py
│     ├── imgSubscriber.py
│     └── ballooon_pos.pth
├── Dockerfile
├── .devcontainer/       # VS Code Remote Containers support
└── README.md
```

## 📦 Setup

### 1. Clone the repo

```bash
git clone https://github.com/jdmwan/balloonpositiondetector.git
cd balloonpositiondetector
```

### 2. Train your own model (optional)

If you'd like to train your own CNN:
```bash
cd pytorchopencv
python balloon_pos.py
```

Model output will be saved as `balloon_pos.pth`.

### 3. Run ROS2 Node and Unity

- On the Unity side, follow the [ROS–Unity Integration Tutorial](https://github.com/Unity-Technologies/ROS-TCP-Connector) to set up and run the **ROS TCP Connector**.
- On the Mac/Windows side, run the Python node that publishes detection data, and use **roslibpy** or [rosbridge_server](https://github.com/RobotWebTools/rosbridge_suite) to relay messages. (this is currently how its setup during development)
- Run `rosbridge_server` and `ROS TCP Connector` in the dockerfile provided.

Make sure your ROS2 environment is sourced:

```bash
source /opt/ros/foxy/setup.bash
```

Then, run the two packages, `rosbridge_server` and `ROS TCP Connector`, according to their tutorials. Then, you can run the subscriber node on your Mac side.

```bash
cd nodes
python3 imgSubscriber.py
```
## 🎥 Demo

> [!NOTE]  
> Video demo coming soon – shows Unity camera following a moving balloon while ROS2 receives positional updates in real time.

## ⚠️ Notes

- Model weights (`.pth`) are **.gitignored** due to GitHub's 100MB limit. Upload your own or contact me for access.
- Unity project not required to test core Python/ROS functionality.

## 💬 Author

Jeffrey Wan  
[GitHub](https://github.com/jdmwan) • [LinkedIn](https://www.linkedin.com/in/your-link-here/)  

---

## 📄 License

MIT License
