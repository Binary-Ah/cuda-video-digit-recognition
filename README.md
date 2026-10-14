# CUDA/cuDNN Video Digit Recognition Pipeline (mnistCUDNN + YOLO)

동영상에서 YOLO로 손글씨 숫자의 위치를 검출하고, 숫자가 **등장한 횟수만큼만** 28×28 PGM으로 저장한 뒤,
cuDNN 기반 `mnistCUDNN` 바이너리로 분류하는 파이프라인입니다. 분류 쪽에서는 `mnistCUDNN.cpp`에 추론 시간 측정과
워밍업을 추가하고 `cudnnFind` 알고리즘 탐색을 제거(IMPLICIT_GEMM 고정)해 추론 시간을 수백 ms에서 한 자릿수 ms로 줄였습니다.

- 2026-1학기 AI가속기 수업 기말 프로젝트 (2026.03.01 ~ 06.10)
- 팀: 오규빈 · 이진아 (2인 공동)

## Problem

- 입력: 손글씨 숫자가 차례로 나타나는 동영상. 같은 숫자가 여러 프레임에 걸쳐 보이므로 프레임마다 저장하면 중복이 생긴다.
- 출력: 숫자가 등장한 횟수만큼의 28×28 PGM 파일과, 각 파일에 대한 cuDNN 분류 결과.
- 제약: mnistCUDNN 샘플의 `readImage`가 읽는 P5 PGM 포맷과 MNIST 전처리 규약(20×20 안에 맞추고 무게중심을 14,14에 정렬)을 지켜야 정확도가 나온다.

## Data

- 입력: 숫자가 등장하는 동영상, YOLO 가중치 `best.pt`(자체 학습).
- mnistCUDNN 샘플 동봉 가중치(`conv1.bin`, `conv2.bin`, `ip1.bin`, `ip2.bin`)로 분류.
- 산출물: `pgm_output/frame_{프레임}_digit_{클래스}_conf_{신뢰도}_{순번}.pgm`.

## Method

### 1. 검출 + 중복 제거 (`run_pipeline.py`)
- ultralytics YOLO `predict(stream=True)`, `imgsz 320`, `vid_stride 3`(3프레임마다 1장), `conf 0.5`.
- 프레임마다 가장 신뢰도 높은 검출 1개만 본다.
- **segment 기반 중복 제거**: 이전 검출과 `gap`(기본 2프레임) 이상 끊기거나 박스 폭의 `move`(기본 2.0)배 이상 위치가 점프하면 새 숫자로 판정. 같은 segment 안에서는 conf가 가장 높은 프레임 1장만 보관.
- `min_hits 3` 미만인 segment는 깜빡임/오검으로 버림.

### 2. MNIST 전처리 (`to_mnist_28`)
- Grayscale → Gaussian blur(3×3) → OTSU 이진화 → 글씨가 흰색이 되도록 반전.
- 글씨 영역을 타이트하게 자른 뒤 긴 변을 20 px로 리사이즈(INTER_AREA) → 28×28 캔버스 중앙 배치 → 무게중심이 (14,14)에 오도록 정수 시프트.
- P5(binary) PGM으로 저장.

### 3. cuDNN 분류 (`mnistCUDNN.cpp`, 수정본)
- CUDA event로 샘플당 추론 시간 측정.
- `cudnnFindConvolutionForwardAlgorithm` 탐색 제거 → `IMPLICIT_GEMM` 고정.
- 첫 호출 워밍업 추가 후 본 측정.
- 파이프라인 끝에서 `./mnistCUDNN dir=<pgm_output>`를 자동 호출.

## Results


## My role

- 2인 공동 프로젝트(오규빈·이진아): 영상 검출 파이프라인(`run_pipeline.py`)과 cuDNN 추론부 수정.

## How to run

```bash
# 의존성
pip install ultralytics opencv-python numpy

# 1. cuDNN 분류기 빌드 (CUDA 샘플 mnistCUDNN 디렉터리에서)
make

# 2. 영상 → PGM 저장 → 분류까지 한 번에
python run_pipeline.py --video input.mp4
#   주요 옵션: --conf 0.5 --imgsz 320 --vid_stride 3 --min_hits 3 --gap 2 --move 2.0
#              --invert (잉크가 희미한 영상) --no_classify (PGM만 저장) --mnist ./mnistCUDNN
# 출력: pgm_output/ (매 실행마다 초기화됨)
```

## Files

```
README.md
run_pipeline.py        # 검출 → segment 중복 제거 → MNIST 전처리 → PGM 저장 → mnistCUDNN 호출
.gitignore
mnistCUDNN.cpp         # cuDNN LeNet 추론부 (추론 시간 측정·cudnnFind 제거·워밍업 수정본)
Makefile               # NVIDIA cuDNN 샘플 기반 빌드 파일
eval_rule.py           # 저장된 PGM을 대비 보정 후 mnistCUDNN으로 분류, 정확도·추론 시간 집계
```

`best.pt`, `*.bin` 가중치, 영상, `pgm_output/`은 커밋하지 않습니다.
