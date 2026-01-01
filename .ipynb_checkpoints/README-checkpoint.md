본 프로젝트는 Real-Time Traffic Density Estimation with YOLOv8을 참고하여 구현하였습니다. 
출처: [Real-Time Traffic Density Estimation with YOLOv8](https://www.kaggle.com/code/farzadnekouei/real-time-traffic-density-estimation-with-yolov8/notebook) by Farzad Nekouei (Kaggle)

## 프로젝트 진행 흐름

    1. COCO 데이터셋으로 사전학습된 YOLOv8 모델을 사용하여 교통 영상 이미지에 대해 초기 예측을 수행함.
<p align="center">
      <img src="Before training.png" width="600"><br>
      <em>Before training</em>
    </p>
    2. 교통 밀도 추정을 목적으로 한 Top-View(탑뷰) 차량 데이터셋을 사용함.
<p align="center">
  <img src="valid/images/2_mp4-5_jpg.rf.bb490ff3835ac898dedea95c973038f5.jpg" width="200">
  <img src="valid/images/6_mp4-1_jpg.rf.a30955a1b5b2a8db39354221db4f5b5f.jpg" width="200">
  <img src="valid/images/6_mp4-29_jpg.rf.e73ca25e92e590b325006f010ad4e319.jpg" width="200">
  <img src="valid/images/11_mp4-28_jpg.rf.afaa0527199e4ac7b9564b9552575d33.jpg" width="200">
</p>

<p align="center">
  <img src="valid/images/test_mp4-15_jpg.rf.2e20880f4bc2ad6347a6a98e8f4ef849.jpg" width="200">
  <img src="valid/images/16_mp4-1_jpg.rf.3493f4b7618e207609847857a20dbaff.jpg" width="200">
  <img src="valid/images/12_mp4-10_jpg.rf.4bb699a2ec90e19cb4680ee239ae579c.jpg" width="200">
  <img src="valid/images/7_mp4-6_jpg.rf.3f9214d14313fa7dd572e7739bbe7398.jpg" width="200">
</p>

<p align="center">
  <em>Before Training – Validation Images (Top-view Traffic Scenes)</em>
</p>
    3. 해당 데이터셋을 기반으로 YOLOv8 모델을 추가 훈련함.
    4. 기존 COCO 사전학습 모델로 예측했던 동일 이미지에 대해, 학습된 모델로 다시 예측을 수행하여 성능 변화를 비교함.
    5. 학습 후에 성능이 좋아지긴 했지만 일부 원거리 차량 및 작은 객체 검출 한계가 존재하여, 성능 향상을 위해 다음과 같은 개선을 시도함.
       - 추론 시 입력 이미지 해상도 확장
       - 고해상도 이미지에 대한 타일 기반 추론 적용
       - GPU 제약 범위 내에서 훈련 이미지 크기 확장
    6. 위 개선 과정을 거쳐 학습된 최적의 모델(best model)을 사용하여 동영상 추론을 수행함.
    7. 동영상 추론 결과를 기반으로, 관심 영역(ROI)을 통과하는 차량을 Tracking ID를 이용해 누적 집계하여 교통량을 산출함.

## 추가/수정한 부분

1. 추론 이미지 사이즈 확장
    - 기본 640 → 1960 으로 조정하여 원거리 객체 탐지 성능 개선 시도함.

2. 훈련 이미지 사이즈 확장
    - 제한된 GPU 환경에서 가능한 범위 내에서 훈련 이미지 크기를 확장함.
        - 기본 512 → 640 으로 조정하여 작은 객체 탐지 성능 개선 시도함.
    - batch 16에서 2로 조정함으로써 학습 속도 크게 느려짐(epoch당 step ↑)
    - gradient 안정성이 떨어져 loss curve가 요동칠 수 있음.

3. 타일 추론 실험
    - 고해상도 이미지를 타일 단위(8조각)로 나누어 추론하는 방식으로 개선 시도함.

4. 동영상 추론 시 누적 카운트 기능 추가
    - 기존 프레임별 차량 수, 교통 현재 상황이 아닌,
      **차량이 가상의 선을 통과할 때 누적카운트**하는 기능 구현
    - 기존 `predict`을 통해 추론만 했던 것을 `track`을 통해 객체마다 Tracking ID 추적하는 기능 구현
    - ID에 좌표를 부여해 기존 좌표와 비교하여 차량 수 집계

5. 주석 한글화
    - 코드 전체 주석을 한글로 바꿔 가독성 높임.

## 메모

1. Tracking ID 원리

    - **track 원리**
        1. 첫 번째 차량 등장
            - YOLO + `track()`→ 새로운 객체 발견 → `tid = 1` 할당
            - 처음이라 `prev_x`에 `1`키가 없음 → 조건문(`if tid in prev_x`) 스킵
            - 마지막에 저장(`prev_x[1] = (cx, cy)`)

        2. 두 번째 차량 등장
            - 새로운 객체라 `tid = 2` 할당
            - 조건문 스킵
            - 마지막에 저장(`prev_x[2] = (cx, cy)`)

        3. 이후 프레임에서 같은 차량들 다시 등장
            - YOLO 추적 계속 이어지므로
                - tid = 1 → 여전히 tid = 1
                - tid = 2 → 여전히 tid = 2
            - 조건문(`if tid in prev_x`) 만족
            - 이전 좌표 `px, py = prev_x[tid]`에 저장 가능
            - 이전 좌표 `px, py`와 현재 좌표 `cx, cy`비교 → 라인 넘었는지 여부 판단


