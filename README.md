# FALL_DETECTION Using a Webcam (실시간 낙상 추론)

## 0. 폴더 설명
<pre>
project
├── data: 학습 데이터
│   ├── falling_video: 낙상 영상
│   ├── normal_video: 보행 영상
│── dataset:전처리 데이터 및 코드
│   ├──saved_pkl:전처리 데이터 저장
│── predic_video:낙상 결과 추론 코드(시각화)
│── stgcn:
│   ├──config:custom yaml
│   ├──model:학습 모델
│   ├──net:stgcn.py
│── webcam : 웹캠 실시간 낙상추론 코드(skeleton 시각화)
</pre>

## 1. 초기 세팅 
### 모델 학습시에는 GPU 가상 서버에서 구동하는것을 권장
### 실행 경로는 사용자 설정에 맞게 수정해서 사용할 것
- ST-GCN 다운로드 및 초기 설정
- 방법 : git clone https://github.com/open-mmlab/st-gcn.git

## 2. 학습 데이터 다운로드
### AI-HUB 사이트에서 타 낙상 video 다운 가능
- 낙상 데이터 다운로드, 타 낙상 데이터 사용시 make_clip.py 활용한 원본영상 NORAML,FALL분류 (30FPS 권장)

## 3. 데이터 전처리 시행 
- 모델 학습에 필요한 피클 파일 생성 및 원본 video 넘파이 파일 생성
- shuffle.py 결과는 검증 데이터로 사용됨
- COCO모델과 동일한 관절 추출하기 위해 보간법으로 처리
- 방법:
     1. convert_pkl_mediapipe.py 실행
     2. shuffle.py 실행
     3. convert_pkl_origin_mediapipe.py 실행

## 4. 모델 훈련 및 추론 수행
### STGCN 초기 설정 후 수행할 것 가
-모델 학습 및 추론 정확도 검증 타 추론 영상 생성시 3-3 코드 활용할 것
-방법:
    1. custom yaml 생성 
    2. st-gcn 루트 경로에서 cmd 명령어 실행
    3. prediction_accu.py 실행
    4. prediction_video.py 실행
    
## 5. 웹캠을 활용한 실시간 추론
### 로컬에서 수행시 깃허브 st-gcn 디렉토리 다운로드 후 실행할 것
-Flask 서버를 사용하여 추론 실시간 스트리밍 
-방법:
    1. 가상서버 혹은 저장된 모델 다운로드
    2. cam_server.py 실행
    3. 브라우저에서 낙상 추론 스트리밍 확인
    









