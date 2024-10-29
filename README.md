[Jae-Min Cho, Cho, Deun-Sol, and Won-Tae Kim. "Guided Deep Reinforcement Learning Framework Using Automated Curriculum Scheme for Accurate Motion Planning." Available at SSRN 4848297.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4848297)<br><br>
# 🤖 Guided Deep Reinforcement Learning Framework Using Automated Curriculum Scheme for Accurate Motion Planning
![Title](https://github.com/user-attachments/assets/5429e937-7ee4-4db0-82c0-d7888ee563f8)<br><br>

![Title2](https://github.com/user-attachments/assets/d6cb27a6-7fdc-4aaf-8366-89b444d08f34)<br><br>

## 💻 Operating System Installation

Refer to the [guide here](https://blog.naver.com/jm_0820/223001100698) for operating system installation.

## 🛠️ ROS Installation

Refer to the [instructions here](http://wiki.ros.org/noetic/Installation/Ubuntu) for installing ROS Noetic.

## 🦾 Moveit Installation

Install Moveit with the following commands:

```bash
sudo apt install ros-noetic-moveit
sudo apt-get install ros-noetic-joint-trajectory-controller
sudo apt-get install ros-noetic-rosbridge-server
```

## 📁 ROS Workspace Setup

Refer to the [guide here](http://wiki.ros.org/ko/catkin/Tutorials/create_a_workspace) for setting up the ROS workspace.

---------------------------------------------------------

## ⚙️ Options

### 📅 System Update

```bash
sudo apt-get update
sudo apt-get upgrade
```

### ⌨️ Korean Keyboard Setup

Refer to the [Korean keyboard setup guide](https://shanepark.tistory.com/231).

### 🐍 pip Installation

```bash
sudo apt-get install python3-pip
```

### 💻 Additional Software Installation

You can install additional software via the following links:

- [GitHub Desktop](https://gist.github.com/berkorbay/6feda478a00b0432d13f1fc0a50467f1)
- [TeamViewer](https://www.teamviewer.com/ko/download/linux/)
- [VScode](https://code.visualstudio.com/download)

```bash
# Install KVM switch software (barrier)
sudo apt install barrier -y

# Install an enhanced terminal (terminator)
sudo apt-get install terminator
```

---------------------------------------------------------

## 🎨 Graphic Driver, CUDA, and cuDNN Installation

### 🚮 Remove Existing Graphics Driver

```bash
sudo apt --purge remove *nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*
```

### 🎯 Install Graphics Driver

```bash
# Check available drivers
ubuntu-drivers devices

# Select and install the version
sudo apt-get install nvidia-driver-(Version, ex 470)
sudo apt-get install dkms nvidia-modprobe

sudo apt-get update
sudo apt-get upgrade

sudo reboot now

# Verify driver installation and check recommended CUDA version
nvidia-smi
```

### 🖥️ CUDA Installation (Recommended 11.8 or 12.1)

Install after checking [compatibility between GPU Driver and CUDA version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id4).

```bash
sudo apt install nvidia-cuda-toolkit
```

Refer to the [CUDA installation guide](https://developer.nvidia.com/cuda-toolkit-archive) for installation.<br/><br/>
Among the installation options, it is recommended to choose "runfile (local)," grant chmod 777 permission, and then execute.

```bash
nvcc -V
# If the version does not appear, refer to "bash convenience configuration" 1
```

### 💾 cuDNN Installation

Install after checking [cuDNN version compatibility](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).

Refer to the [cuDNN installation guide](https://developer.nvidia.com/rdp/cudnn-archive).<br/><br/>
Recommended deb format file such as "Local Installer for Ubuntu20.04 x86_64 (Deb)"

```bash
sudo apt update

# If an error occurs
sudo rm /etc/apt/sources.list.d/cuda*
sudo rm /etc/apt/sources.list.d/cudnn*
```

### 🔥 PyTorch Installation (Python 3.9 or later recommended)

Refer to the [CUDA-compatible PyTorch installation guide](https://pytorch.org/get-started/locally/).<br/><br/>
Run the code below to check CUDA and cuDNN recognition:

```python
import torch

print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

print(torch.backends.cudnn.enabled)
print(torch.backends.cudnn.version())
```

---------------------------------------------------------

## 🦾 Niryo Ned2 (Robot Arm) ROS Package Download

Download the package from the [shared ROS package link](https://drive.google.com/file/d/1asuf5u0nxEIL4igmGXXH0zTojgIlM7af/view?usp=sharing).

```bash
# Extract and place it in ~/catkin_ws/src
cd ~/catkin_ws
catkin_make
source ./devel/setup.bash
source ~/.bashrc
roslaunch ned2_moveit demo_gazebo.launch
```

---------------------------------------------------------

## 🛠️ bashrc Convenience Configuration

Edit the bashrc file using `gedit ~/.bashrc`, and add the following lines at the bottom:

```bash
# Specify CUDA path
# Check installed CUDA with cd /usr/local and ls
export PATH=/usr/local/cuda-(your CUDA version)/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-(your CUDA version)/lib64:$LD_LIBRARY_PATH

# Adjust to use only Python 3.x
alias python=python3
alias pip=pip3

# ROS setup
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

# ROS shortcuts
alias sb="source ~/.bashrc"
alias cm="catkin_make & source ./devel/setup.bash"
alias rc='rosclean purge -y'
alias rn='rosclean purge -y & roslaunch ned2_moveit demo_gazebo.launch'

# Specify ROS IP and port to avoid overlap on the same local network
# Check your IP with ifconfig
export ROS_MASTER_URI=http://(your IP):(port number you want to use, default = 11311)
# example) export ROS_MASTER_URI=http://192.168.0.121:11311
export ROS_HOSTNAME=(your IP)
# example) export ROS_HOSTNAME=192.168.0.121
```
