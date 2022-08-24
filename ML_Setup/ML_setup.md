# 머신러닝 환경설치



## 1. Conda 설치

- 링크 : https://www.anaconda.com/products/distribution



## 2. Conda 가상환경 생성

1. ``conda create -n 가상환경이름 python=3.7(원하는버전)``
2. ``conda env list``로 가상환경 확인
3. ``conda activate 가상환경이름``으로 해당 가상환경으로 진입
   - (``conda deactivate``명령어로 해제가능)



## 3. Cuda 환경 설정

**윈도우 환경 기준**

1. 그래픽카드에 맞는 Cuda 버전 확인
   - 링크 : https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
   - RTX3060 : compute capability 8.6 => CUDA SDK 11.1 - 11.4
   - CUDA 버전 선택 : 11.4
2. Visual Studio 다운로드
   - 2019 Version : https://docs.microsoft.com/ko-kr/visualstudio/releases/2019/release-notes
   - 확장없이 기본으로 다운로드
   - Visual Studio가 설치되어있다면 Pass
3. CUDA 다운로드
   - 링크 : https://developer.nvidia.com/cuda-toolkit-archive
   - 필자는 11.4.1버전 다운로드
4. cuDNN 다운로드
   - CUDA 버전에 맞는 cuDNN을 다운받아야 한다.
   - 링크 : https://developer.nvidia.com/rdp/cudnn-archive
   - (회원가입을 해야 다운로드가 가능하다.)
   - 필자는 cuDNN 8.4.1버전을 다운받았다.
   - 다운로드 받은 파일의 압축을 풀어서 'cuda'안에 있는 파일들을 ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4``에 다 붙여넣어야한다. v11.4는 버전별로 폴더의 이름이 다르다.
5. 설치확인
   - cmd 창에 ``nvcc --version``, ``nvidia-smi``의 명령어를 쳤을 때, 설치한 CUDA의 버전이 표기되면 설치가 완료된 것이다.



## 4. Conda Package 설치

- Tensorflow와 pytorch같이 GPU를 이용하는 패키지들은 CUDA의 버전과 호환이 되도록 설치해야한다. 
- 하지만 일일이 버전을 확인하기에는 너무 번거롭기 때문에 모든 패키지를 conda-forge로 다운로드 받으면 호환성에 문제 없이 다운로드가 가능하다.

- 예시 : ``conda install -c conda-forge dlib``



### 1) Tensorflow 설치

- ``conda install -c conda-forge tensorflow-gpu``



### 2) pytorch 설치

- ``conda install pytorch torchvision torchaudio cudatoolkit=11.4 -c pytorch -c conda-forge``



### 3) konlpy 설치

1. JAVA 설치(JDK)

   - 링크 : https://www.oracle.com/java/technologies/downloads/
   - 시스템 환경변수 설정
     - 변수이름 : JAVA_HOME
     - 변수 값 : C:\Program Files\Java\jdk-18.0.2.1

2. jpype 설치

   - 링크 : https://www.lfd.uci.edu/~gohlke/pythonlibs/#jpype

   - 설치한 파이썬의 버전에 맞게 다운로드

   - 필자는 3.7버전으로 환경을 구성했기 때문에``Jpype1-1.1.2-cp37m-win_amd64.whl``을 다운받음.

   - ![image-20220825045740972](C:\Users\Park Ji Seong\AppData\Roaming\Typora\typora-user-images\image-20220825045740972.png)

     (이미지는 jupyter notebook의 터미널이다. 가상환경에 konlpy를 설치하기 위함)

   - 해당 파일이 있는 위치로 이동해서``pip install 1.1.2-cp37m-win_amd64.whl``을 통해서 설치

3. konlpy 설치

   - ``pip install konlpy``로 설치한다.



### 4) Mecab 설치

- 참고 : https://cleancode-ws.tistory.com/97

- 설치 후 사용은 ``mecab('C:\mecab\mecab-ko-dic')``로 사용한다.

