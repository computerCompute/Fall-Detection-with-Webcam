# FALL_DETECTION (웹캠 추론 추가)

## 0. 폴더 설명
<pre>
project
├── data: 학습 데이터
│   ├── falling_video: 낙상 영상
│   ├── normal_video: 보행 영상
│── dataset:전처리 데이터 및 코드
│   ├──saved_pkl:전처리 데이터 저장
│── predict_video:낙상 결과 추론 코드(시각화)
│── stgcn:
│   ├──config:custom yaml
│   ├──model:학습 모델
│   ├──net:stgcn.py
│── webcam : 웹캠 실시간 낙상추론 코드(skeleton 시각화)
</pre>
학습이 완료된 모델을 활용해 추론만 진행하려는 사용자는 5단계부터 시작하면 됩니다.

## 1. 초기 세팅 
### 모델 학습시에는 GPU 가상 서버에서 구동하는 것을 권장합니다
### 실행 경로는 사용자 환경에 맞게 수정해야 합니다
- ST-GCN 다운로드 및 초기 설정
- 방법 : 'git clone https://github.com/open-mmlab/st-gcn.git'

## 2. 학습 데이터 다운로드
### AI-HUB 사이트에서 낙상 영상 다운로드 가능
- make_clip.py를 활용해 NORMAL/FALL로 분류

## 3. 데이터 전처리 시행 
- 학습용 .pkl 파일 및 원본 영상의 NumPy 파일 생성
- shuffle.py 결과는 검증 데이터로 사용됨
- COCO 모델 기준 관절 수에 맞추기 위해 일부 관절은 보간 처리
- 방법:
     1. convert_pkl_mediapipe.py 실행
     2. shuffle.py 실행
     3. convert_pkl_origin_mediapipe.py 실행

## 4. 모델 훈련 및 추론 수행
### STGCN 초기 설정 후 실행 할 것 
- 모델 학습과 추론 정확도 검증 후, 타 영상에 대한 추론 결과를 생성할 경우 3-3 코드를 활용하세요
- 방법:
     1. custom_config.yaml 생성 
     2. st-gcn 루트 경로에서 cmd 명령어 실행
     3. prediction_accu.py 실행
     4. prediction_video.py 실행
    
## 5. 웹캠을 활용한 실시간 추론
### 실시간 추론 스트리밍을 로컬에서 실행하려면, GitHub에 있는 `/fall_detection/st-gcn` 디렉토리 다운로드하여 실행해야 합니다
- Flask 서버를 활용한 실시간 스트리밍  
- 방법:
     1. 가상서버 혹은 저장된 모델 다운로드
     2. cam_server.py 실행
     3. 브라우저에서 아래 주소로 접속해 실시간 추론 확인 (http://localhost:####/video)

