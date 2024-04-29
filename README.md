# Soft-Actor-Critic-for-Robotarm

## 운영체제 설치
https://blog.naver.com/jm_0820/223001100698

## ROS 설치
http://wiki.ros.org/noetic/Installation/Ubuntu

## Moveit 설치
    sudo apt install ros-noetic-moveit
    sudo apt-get install ros-noetic-joint-trajectory-controller
    sudo apt-get install ros-noetic-rosbridge-server

## ROS 작업공간 설정
    http://wiki.ros.org/ko/catkin/Tutorials/create_a_workspace

---------------------------------------------------------

## 옵션

### 시스템 업데이트
    sudo apt-get update
    sudo apt-get upgrade

### 한국어 키보드 설정
    https://shanepark.tistory.com/231

### pip 설치
    sudo apt-get install python3-pip

### 추가 프로그램 설치
    #### GitHub Desktop 설치
        https://gist.github.com/berkorbay/6feda478a00b0432d13f1fc0a50467f1

    #### KVM 스위치 소프트웨어 (barrier) 설치
        sudo apt install barrier -y

    #### 향상된 터미널 (terminator) 설치
        sudo apt-get install terminator

    #### TeamViewer 설치
        https://www.teamviewer.com/ko/download/linux/
        cd Download
        dpkg -i (package name)

    #### VScode 설치
        https://code.visualstudio.com/download
        cd Download
        dpkg -i (package name)

    #### Python3 명령어 변경
        gedit ~/.bashrc
        < add to bottom >
        alias python=python3
        alias pip=pip3

---------------------------------------------------------

## 그래픽 드라이버 및 CUDA 및 cuDNN 설치

### 기존에 설치된 그래픽 드라이버 제거
    sudo apt-get purge nvidia*
    sudo apt-get autoremove
    sudo apt-get autoclean
    sudo rm -rf /usr/local/cuda*

### 그래픽 드라이버 설치
    #설치 가능한 드라이버 확인
    ubuntu-drivers devices
    
    #버전 선택 후 설치
    sudo apt-get install nvidia-driver-(Veirsion, ex 470)
    sudo apt-get install dkms nvidia-modprobe
    
    sudo apt-get update
    sudo apt-get upgrade
    
    sudo reboot now

    #그래픽드라이버 설치 확인 및 추천 CUDA 버전 확인
    nvidia-smi

### CUDA 설치
    #CUDA 버전 선택 후 설치
    #GPU Driver와 CUDA 버전 호환성 확인 : https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id4
    
    sudo apt install nvidia-cuda-toolkit
    https://developer.nvidia.com/cuda-toolkit-archive
    
    nvcc -V
    #만약 버전이 안나온다면 "bash 편의설정" 1 참조
    
### cuDNN 설치
    #cuDNN 버전 호환성 확인 : https://en.wikipedia.org/wiki/CUDA#GPUs_supported
    https://developer.nvidia.com/rdp/cudnn-archive
    
---------------------------------------------------------

## bashrc 편의설정
    gedit ~/.bashrc
    
    ### 맨 아래에 원하는 라인을 추가
    
    ### CUDA 경로 지정 (윈도우의 시스템 환경 변수와 같음)
    export PATH=/usr/local/cuda-12.4/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

    ### python 3.x버전만 사용하도록 조정
    alias python=python3
    alias pip=pip3

    ### ros setup
    source /opt/ros/noetic/setup.bash
    source ~/catkin_ws/devel/setup.bash

    ### ros 단축어 설정
    alias sb="source ~/.bashrc"
    alias cm="cd ~/catkin_ws & catkin_make"
    alias rc='rosclean purge -y'
    alias rn='rosclean purge -y&roslaunch ned2_moveit demo_gazebo.launch'

    ### ros IP 지정, 같은 로컬 네트워크에서 서로 겹치지 않게하는 역할
    export ROS_MASTER_URI=http://192.168.1.121:11313
    export ROS_HOSTNAME=192.168.1.121
