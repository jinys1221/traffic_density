"""
이 파이썬 스크립트는 YOLO Detecion 데이터셋에 존재하지 않는 '도로(Road)' 클래스를 반자동으로 생성하기 위한 보조 스크립트입니다.

- 기존 : Vehicle 클래스만 라벨에 존재
- 추가 : Road 클래스
- SAM2, ROI 마스크, OpenCV 전처리 등을 활용하여 도로 영역 라벨 자동화
"""

# import
import os
from pathlib import Path
import glob
import cv2
import numpy as np
import torch

# 경로 설정 
BASE = r"D:/project/traffic density/segmentation"   # seg_train / seg_valid
CFG_PATH = r"D:/project/traffic density/segmentation/sam2/sam2/configs/sam2/sam2_hiera_b+.yaml"
CKPT_PATH = r"D:/project/traffic density/segmentation/sam2/checkpoints/sam2_hiera_base_plus.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAM2 로드
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_model = build_sam2(CFG_PATH, CKPT_PATH, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)


# ==== 유틸: 마스크 → YOLO-seg(.txt) 라인 저장 ================================
def save_mask_as_yolo(mask, txt_path: Path, class_id=1, min_points=4):
    H, W = mask.shape[:2]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if len(cnt) < min_points:
            continue
        pts = []
        for x, y in cnt.reshape(-1, 2):
            pts.extend([x / W, y / H])
        lines.append(str(class_id) + " " + " ".join(f"{p:.6f}" for p in pts))

    # 기존 라벨(차량 등)이 있으면 유지 + 도로 라벨 추가
    existing = []
    if txt_path.exists():
        existing = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for ln in existing:
            f.write(ln + "\n")
        for ln in lines:
            f.write(ln + "\n")

# ==== 배치 처리: 한 이미지 → SAM2 예측 → 도로 라벨 추가 =====================
def process_image(img_path: Path, labels_dir: Path, save_mask_png=False, masks_dir: Path | None = None, class_id=1):
    img_bgr = cv2.imread(str(img_path))
    assert img_bgr is not None, f"이미지 읽기 실패: {img_path}"
    H, W = img_bgr.shape[:2]

    # SAM2 입력 세팅
    predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # (오토 프롬프트) 도로 가정: 하단 40% 영역을 덮는 바운딩박스
    y1 = int(H * 0.60)
    x1, x2 = 0, W - 1
    y2 = H - 1
    box = np.array([x1, y1, x2, y2])

    masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
    # 가장 큰 마스크 선택 (도로는 넓은 편이라는 가정)
    areas = [int(m.sum()) for m in masks]
    mask = masks[int(np.argmax(areas))].astype(np.uint8) * 255

    # (선택) 마스크 jpg로 저장
    if save_mask_png and masks_dir is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)
        out_jpg = masks_dir / (img_path.stem + "_road.jpg")
        cv2.imwrite(str(out_jpg), mask)

    # YOLO-seg txt 저장/갱신 (차량 라벨 유지 + 도로 추가)
    txt_path = labels_dir / (img_path.stem + ".txt")
    save_mask_as_yolo(mask, txt_path, class_id=class_id)

# ==== train / valid 둘 다 실행 ==============================================
def run_all():
    splits = ["seg_train", "seg_valid"]
    for sp in splits:
        img_dir   = Path(BASE) / sp / "images"
        labels_dir= Path(BASE) / sp / "labels"
        masks_dir = Path(BASE) / sp / "masks"   # 마스크 PNG 보관(선택)

        labels_dir.mkdir(parents=True, exist_ok=True)  # YOLO 라벨 저장 위치
        masks_dir.mkdir(parents=True, exist_ok=True)

        img_paths = []
        for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp"):
            img_paths.extend(glob.glob(str(img_dir / ext)))
        img_paths = [Path(p) for p in sorted(img_paths)]

        print(f"[{sp}] 이미지 {len(img_paths)}장 처리 시작")
        for i, p in enumerate(img_paths, 1):
            try:
                process_image(p, labels_dir, save_mask_png=True, masks_dir=masks_dir, class_id=1)
            except Exception as e:
                print(f"  - 실패({p.name}): {e}")
            if i % 50 == 0:
                print(f"  ... {i}장 완료")

        print(f"[{sp}] 처리 완료\n")

if __name__ == "__main__":
    run_all()