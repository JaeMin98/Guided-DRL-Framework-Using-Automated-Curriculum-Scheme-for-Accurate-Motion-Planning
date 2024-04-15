# Soft-Actor-Critic-for-Robotarm

## 운영체제 설치
https://blog.naver.com/jm_0820/223001100698

## ROS 설치
http://wiki.ros.org/noetic/Installation/Ubuntu

## Moveit 설치
    sudo apt install ros-noetic-moveit
    sudo apt-get install ros-noetic-joint-trajectory-controller
    sudo apt-get install ros-noetic-rosbridge-server
    sudo apt-get install ros-noetic-joint-trajectory-controller

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

### 설치를 위한 Key 값 정의
    sudo wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
    sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

### 그래픽 드라이버 설치 (예: 470)
    sudo apt-get install nvidia-driver-470
    sudo apt-get install dkms nvidia-modprobe
    sudo apt-get update
    sudo apt-get upgrade
    sudo reboot now
    nvidia-smi

### CUDA 설치 (예: 11.4)
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb
    sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
    sudo apt-get update
    sudo apt-get -y install cuda
    ls /usr/local | grep cuda
    sudo sh -c "echo 'export PATH=$PATH:/usr/local/cuda-11.4/bin'>> /etc/profile"
    sudo sh -c "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/lib64'>> /etc/profile"
    sudo sh -c "echo 'export CUDARDIR=/usr/local/cuda-11.4'>> /etc/profile"
    source /etc/profile
    nvcc -V

### cuDNN 설치 (8.2.2 for CUDA 11.4, cuDNN Library for Linux (X86_64))
    cuDNN 소스파일 다운로드 (8.2.2 for CUDA 11.4, cuDNN Library for Linux (X86_64))
    https://developer.nvidia.com/rdp/cudnn-archive
    cd Downloads/
    tar xvzf cudnn-11.4-linux-x64-v8.2.2.26.tgz
    sudo cp cuda/include/cudnn* /usr/local/cuda/include
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda-11.4/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    sudo ln -sf /usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.2.2 /usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
    sudo ln -sf /usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.2.2  /usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
    sudo ln -sf /usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.2.2  /usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
    sudo ln -sf /usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.2.2  /usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
    sudo ln -sf /usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.2.2  /usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
    sudo ln -sf /usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.2.2 /usr/local/cuda-11


https://teddylee777.github.io/linux/ubuntu2004-cuda-update/   
https://lapina.tistory.com/131
