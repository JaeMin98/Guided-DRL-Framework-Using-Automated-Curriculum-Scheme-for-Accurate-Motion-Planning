
# 🤖 Soft-Actor-Critic for Robot Arm

본 README는 ROS 기반 로봇 팔 제어를 위해 Soft-Actor-Critic 알고리즘을 사용하는 방법에 대해 설명합니다. 여기에는 운영체제 설치부터 각종 필수 소프트웨어 설치, 그리고 ROS 작업공간 설정 및 편의 설정에 이르기까지의 과정을 포함하고 있습니다.

## 💻 운영체제 설치

운영체제 설치 가이드는 [여기](https://blog.naver.com/jm_0820/223001100698)를 참고하십시오.

## 🛠️ ROS 설치

ROS Noetic 설치 방법은 [여기](http://wiki.ros.org/noetic/Installation/Ubuntu)에서 확인할 수 있습니다.

## 🦾 Moveit 설치

다음 명령어를 사용하여 Moveit을 설치하십시오:

```bash
sudo apt install ros-noetic-moveit
sudo apt-get install ros-noetic-joint-trajectory-controller
sudo apt-get install ros-noetic-rosbridge-server
```

## 📁 ROS 작업공간 설정

ROS 작업공간 설정 방법은 [여기](http://wiki.ros.org/ko/catkin/Tutorials/create_a_workspace)를 참고하십시오.

---------------------------------------------------------

## ⚙️ 옵션

### 📅 시스템 업데이트

```bash
sudo apt-get update
sudo apt-get upgrade
```

### ⌨️ 한국어 키보드 설정

[한국어 키보드 설정 가이드](https://shanepark.tistory.com/231)를 참고하십시오.

### 🐍 pip 설치

```bash
sudo apt-get install python3-pip
```

### 💻 추가 프로그램 설치

아래 링크를 통해 추가 프로그램을 설치할 수 있습니다:

- [GitHub Desktop](https://gist.github.com/berkorbay/6feda478a00b0432d13f1fc0a50467f1)
- [TeamViewer](https://www.teamviewer.com/ko/download/linux/)
- [VScode](https://code.visualstudio.com/download)

```bash
# KVM 스위치 소프트웨어 (barrier) 설치
sudo apt install barrier -y

# 편의성이 향상된 터미널 (terminator) 설치
sudo apt-get install terminator
```

---------------------------------------------------------

## 🎨 그래픽 드라이버 및 CUDA 및 cuDNN 설치

### 🚮 기존에 설치된 그래픽 드라이버 제거

```bash
sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*
```

### 🎯 그래픽 드라이버 설치

```bash
# 설치 가능한 드라이버 확인
ubuntu-drivers devices

# 버전 선택 후 설치
sudo apt-get install nvidia-driver-(Version, ex 470)
sudo apt-get install dkms nvidia-modprobe

sudo apt-get update
sudo apt-get upgrade

sudo reboot now

# 그래픽 드라이버 설치 확인 및 추천 CUDA 버전 확인
nvidia-smi
```

### 🖥️ CUDA 설치 (11.8 혹은 12.1 설치 권장)

[GPU Driver와 CUDA 버전 호환성 확인](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id4) 후 설치합니다.

```bash
sudo apt install nvidia-cuda-toolkit
```

[CUDA 설치 가이드](https://developer.nvidia.com/cuda-toolkit-archive)를 참고하여 설치합니다.

```bash
nvcc -V
# 만약 버전이 나오지 않는다면 "bash 편의설정" 1 참조
```

### 💾 cuDNN 설치

[cuDNN 버전 호환성 확인](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) 후 설치합니다.

[cuDNN 설치 가이드](https://developer.nvidia.com/rdp/cudnn-archive)를 참고하십시오.

### 🔥 PyTorch 설치

[CUDA 호환 PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)를 참고하여 설치합니다. 아래 코드를 실행하여 CUDA와 cuDNN 인식 여부를 확인합니다:

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

## 🦾 Niryo Ned2 (Robot Arm) ROS 패키지 다운로드

[ROS 패키지 공유 링크](https://drive.google.com/file/d/1R_Lr5dDcLRc0oqfuJMkV8asckWMMeXkj/view?usp=sharing)에서 패키지를 다운로드합니다.

```bash
# 압축을 풀고 ~/catkin_ws/src에 넣기
cd ~/catkin_ws
catkin_make
source ./devel/setup.bash
source ~/.bashrc
roslaunch ned2_moveit demo_gazebo.launch
```

---------------------------------------------------------

## 🛠️ bashrc 편의설정

`gedit ~/.bashrc` 명령어로 bashrc 파일을 편집하고, 맨 아래에 다음 라인을 추가합니다:

```bash
# CUDA 경로 지정
# 설치된 CUDA는 cd /usr/local에서 ls로 확인 가능
export PATH=/usr/local/cuda-(자신의 쿠다 버전)/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-(자신의 쿠다 버전)/lib64:$LD_LIBRARY_PATH

# python 3.x버전만 사용하도록 조정
alias python=python3
alias pip=pip3

# ROS setup
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

# ROS 단축어 설정
alias sb="source ~/.bashrc"
alias cm="catkin_make & source ./devel/setup.bash"
alias rc='rosclean purge -y'
alias rn='rosclean purge -y & roslaunch ned2_moveit demo_gazebo.launch'

# ROS IP 및 포트 지정, 같은 로컬 네트워크에서 서로 겹치지 않게 하는 역할
# ifconfig로 자신의 IP 확인 가능
export ROS_MASTER_URI=http://(자신의 IP):(사용하고자하는 포트번호, default = 11311)
# example) export ROS_MASTER_URI=http://192.168.0.121:11311
export ROS_HOSTNAME=(자신의 IP)
#example) export ROS_HOSTNAME=192.168.0.121
```
