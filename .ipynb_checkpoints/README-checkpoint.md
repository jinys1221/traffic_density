본 프로젝트는 Real-Time Traffic Density Estimation with YOLOv8을 참고하여 구현하였습니다. 
출처: [Real-Time Traffic Density Estimation with YOLOv8](https://www.kaggle.com/code/farzadnekouei/real-time-traffic-density-estimation-with-yolov8/notebook) by Farzad Nekouei (Kaggle)

## 추가/수정한 부분

1. 추론 이미지 사이즈 확장
    - 기본 640 → 1960 으로 조정하여 원거리 객체 탐지 성능 개선 시도함.

2. 훈련 이미지 사이즈 확장
    - 제한된 GPU 환경에서 가능한 범위 내에서 훈련 이미지 크기를 확장함.
        - 기본 512 → 640 으로 조정하여 작은 객체 탐지 성능 개선 시도함.
        - batch 16에서 2로 조정하여 VRAM 오버 방지, 과적합 방지
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

## 기타 개선 사항

1. YOLO 학습 시 결과 저장 경로를 프로젝트 전용 경로로 변경하여 관리함.

YOLO 학습 결과는 기본적으로 C:\Users\jinhyeongsik\runs\detect 경로에 저장됩니다.
본 프로젝트에서는 관리 편의를 위해 D:/project/traffic density/detect 로 이동하여 사용했습니다.

3. Tracking ID 원리를 README에 기술 

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


