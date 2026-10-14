"""
사용법:
  python run_pipeline.py --video input.mp4
  (weights는 같은 폴더의 best.pt, 출력은 pgm_output 폴더로 고정. 매 실행마다 폴더 초기화)
"""

import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


# MNIST 전처리 (정확도의 핵심)
def to_mnist_28(crop_bgr, invert=False):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 글씨가 흰색(255)이 되도록 보정.(반전)
    if not invert:
        bw = 255 - bw

    # 글씨 영역 타이트하게 자르기
    ys, xs = np.where(bw > 0)
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    digit = bw[y0:y1 + 1, x0:x1 + 1]

    # 가장 긴 변을 20으로 리사이즈
    h, w = digit.shape
    if h > w:
        nh, nw = 20, max(1, int(round(w * 20.0 / h)))
    else:
        nw, nh = 20, max(1, int(round(h * 20.0 / w)))
    digit = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)

    # 28x28 캔버스에 일단 가운데 정렬
    canvas = np.zeros((28, 28), dtype=np.uint8)
    oy, ox = (28 - nh) // 2, (28 - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = digit

    # 무게중심으로 미세 시프트
    m = cv2.moments(canvas, binaryImage=False)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        sx, sy = int(round(14 - cx)), int(round(14 - cy))
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        canvas = cv2.warpAffine(canvas, M, (28, 28),
                                flags=cv2.INTER_NEAREST, borderValue=0)
    return canvas


def save_pgm_p5(path, img28):
    """28x28 uint8 -> P5(binary) PGM. mnistCUDNN readImage 호환."""
    h, w = img28.shape
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (w, h))
        f.write(img28.tobytes())


def main():
    ap = argparse.ArgumentParser()
    #디버그용 argument들
    ap.add_argument("--video", required=False, default=None,
                    help="path to input video (required)")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--weights", default=os.path.join(script_dir, "best.pt"))
    ap.add_argument("--out", default="pgm_output")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--min_hits", type=int, default=3,
                    help="이 프레임 수 이상 추적돼야 실수로 판정 (노이즈 제거)")
    ap.add_argument("--invert", action="store_true",
                    help="잉크가 희미한 사진들용")
    ap.add_argument("--imgsz", type=int, default=320,
                    help="추론 해상도. 작을수록 빠름")
    ap.add_argument("--vid_stride", type=int, default=3,
                    help="N프레임마다 1장만 처리.")
    ap.add_argument("--device", default=None,
                    help="'cpu' / '0'(GPU) / None(자동).")
    ap.add_argument("--gap", type=int, default=2,
                    help="검출이 gap만큼 끊기면 다음 숫자로 간주")
    ap.add_argument("--move", type=float, default=2.0,
                    help="박스폭의 N배 이상 위치 점프하면 새 숫자로 간주")
    ap.add_argument("--mnist", default="./mnistCUDNN",
                    help="cuDNN 분류 바이너리 경로")
    ap.add_argument("--no_classify", action="store_true",
                    help="pgm만 저장하고 분류는 건너뛰기")
    args = ap.parse_args()

    if args.video is None:
        ap.print_help()
        print('\nError: --video is required. 예: python run_pipeline.py --video input.mp4')
        return

    # 매 실행마다 출력 폴더를 비우고 시작 (이전 pgm이 섞여 채점 오염되는 것 방지)
    import shutil
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.weights)

    results = model.predict(source=args.video, conf=args.conf,
                            stream=True, verbose=False,
                            imgsz=args.imgsz, vid_stride=args.vid_stride,
                            device=args.device)

    # 시간 구간(segment) 기반 중복 제거 사용
    # 프레임마다 가장 또렷한 검출 1개만 보고, 이전 검출과 시간이 끊기거나(gap) 위치가 확 점프하면 새숫자 판정
    # 같은 segment 안에서는 conf 최고 프레임 1장만 저장.
    segments = []          # 확정된 등장들
    cur = None             # 진행 중인 segment
    last_seen = None       # 마지막으로 검출된 프레임 인덱스
    last_center = None     # 마지막 검출 중심 좌표

    for frame_idx, r in enumerate(results):
        has_det = r.boxes is not None and len(r.boxes) > 0
        if not has_det:
            # 검출 없음: gap이 충분히 벌어지면 현재 등장 종료
            if cur is not None and last_seen is not None \
                    and (frame_idx - last_seen) > args.gap:
                segments.append(cur)
                cur = None
            continue

        frame = r.orig_img
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
        b = int(np.argmax(confs))          # 가장 또렷한 검출 1개
        conf = float(confs[b]); cls = int(clss[b])
        x1, y1, x2, y2 = xyxy[b]
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        box_w = max(1, x2 - x1)

        new_seg = True
        if cur is not None and last_seen is not None:
            gap_ok = (frame_idx - last_seen) <= args.gap
            jump = ((cx - last_center[0]) ** 2 +
                    (cy - last_center[1]) ** 2) ** 0.5
            near = jump <= args.move * box_w   # 박스폭의 배수 이내면 같은 숫자
            if gap_ok and near:
                new_seg = False

        if new_seg:
            if cur is not None:
                segments.append(cur)
            cur = {"conf": conf, "crop": crop.copy(), "cls": cls,
                   "first": frame_idx, "hits": 1}
        else:
            cur["hits"] += 1
            if conf > cur["conf"]:         # 등장 구간 중 가장 또렷한 프레임으로 갱신
                cur["conf"] = conf
                cur["crop"] = crop.copy()
                cur["cls"] = cls

        last_seen = frame_idx
        last_center = (cx, cy)

    if cur is not None:
        segments.append(cur)

    # 깜빡임/오검 제거
    valid = [s for s in segments if s["hits"] >= args.min_hits]
    valid.sort(key=lambda v: v["first"])

    print(f"감지된 등장(=저장될 pgm) 개수: {len(valid)}")
    for seq, rec in enumerate(valid, start=1):
        img28 = to_mnist_28(rec["crop"], invert=args.invert)
        real_frame = rec["first"] * args.vid_stride  # 실제 영상 프레임 번호 근사
        fname = (f"frame_{real_frame:06d}_digit_{rec['cls']}"
                 f"_conf_{rec['conf']:.2f}_{seq:04d}.pgm")
        save_pgm_p5(os.path.join(args.out, fname), img28)
        print("saved:", fname)

    # pgm 다 만든 뒤, 바로 cuDNN(MNIST) 분류까지 실행 (디버그)
    if not args.no_classify:
        import subprocess
        mnist_path = os.path.abspath(args.mnist)
        workdir = os.path.dirname(mnist_path) or "."
        if not os.path.exists(mnist_path):
            print(f"[건너뜀] 분류 바이너리 없음: {mnist_path} "
                  f"(먼저 make 했는지 확인)")
            return
        out_abs = os.path.abspath(args.out)
        print("\n===== MNIST 분류 시작 =====")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env.pop("CUDA_DEVICE_ORDER", None)
        subprocess.run([mnist_path, "dir=" + out_abs], cwd=workdir, env=env)

if __name__ == "__main__":
    main()
