# 파이프라인 개요 (DLStatArb)

이 문서는 저장소의 전체 코드 흐름과 실행/저장 산출물을 자세히 설명합니다.

## 1. 설정 (train_config.json)

- 단일 JSON 파일(`train_config.json`)로 모든 설정을 관리합니다.
- 주요 키
  - 경로
    - `windows_path`: 전처리 윈도우 파일(`.pt`) 저장 위치 (기본 `PROCESSED/residual_windows.pt`).
    - `residuals_path`: 요인모델 잔차(`.parquet`) 저장 위치 (기본 `DATA/residuals.parquet`).
    - `data_dir`: 원천 OHLCV 데이터 디렉토리(옵션). `loader.py`가 사용.
  - 데이터 전처리
    - `lookback`, `stride`, `min_assets`, `min_valid_ratio`, `zero_as_invalid`, `horizon`
  - 분할
    - 날짜 기준: `train_end`, `val_end` (둘 중 하나, 혹은 둘 다)
    - 비율 기준: `train_ratio`, `val_ratio`
  - FactorModel
    - `factor_n_components`(PCA 수), `factor_win_pca`, `factor_win_beta`
  - 학습/장치
    - `batch_size`, `num_workers`, `epochs`, `lr`, `weight_decay`, `grad_clip`, `eval_interval`
    - `objective`(sharpe/meanvar/sqrtMeanSharpe), `weight_scheme`(l1/softmax)
    - `hold_cost`, `device`(mps/cuda/cpu), `dtype`(float32/float16/bfloat16)
    - 혼합정밀은 CUDA에서만 활성, MPS/CPU는 FP32로 동작
  - 체크포인트/출력
    - `checkpoint`: 학습 도중(개선 시) 또는 종료 시(개선 없을 때) 저장되는 체크포인트 경로
    - `history_out`: 학습 히스토리(JSON)
    - `model_out`: 최종 학습 모델의 `state_dict`(pt)
    - `config_out`: 실행에 사용된 해석 완료된 설정(JSON)

## 2. 로더/요인/전처리 (datas.py, factors.py, preprocess.py)

- `datas.py`의 `DataPipeline`이 전체 전처리를 오케스트레이션합니다.
  1. 잔차 보장: `ensure_residuals()`
     - `factors.FactorModel(n_components, win_pca, win_beta, data_dir)` 인스턴스 생성 → `save(out_dir)` 호출
     - `save()`는 내부 `returns` property(= `loader.DataLoader`로 close 로딩 → 정렬 → `pct_change` → inf/NaN 처리)를 사용해 필요 시 `fit_all()`을 수행한 다음, 잔차/설명력(`DATA/*.parquet`) 저장
  2. 윈도우 빌드: `build_windows()`
     - 존재하면 로드, 없으면 `preprocess.WindowBuilder`로 생성 후 `windows_path`에 저장
  3. 분할: `split_windows()`
     - 날짜 기준(`train_end`, `val_end`)이 있으면 `classify.WindowClassifier.by_date()`
     - 없으면 비율 기준으로 `train/val/test` 인덱스를 계산해 분할
  4. 로더 변환: `make_loader()`
     - `Windows` → `ResidualsDataset` → `torch.utils.data.DataLoader`
  5. 실행: `run(batch_size, num_workers, device)`
     - 위 1)~4)를 순서대로 실행하고 `DataArtifacts`로 결과(윈도우, 분할, 로더 3종)를 반환

산출물

- 잔차/설명력: `DATA/residuals.parquet`, `DATA/explained_variances.parquet`
- 윈도우: `PROCESSED/residual_windows.pt`

## 3. 모델 (cnn.py, transformer.py)

- `CNNConfig` + `TransformerConfig`로 구성된 `StatArbModel`.
- `main.py`의 `build_model(cfg)`에서 JSON 설정을 읽어 `ModelConfig`를 생성.

## 4. 학습 루프 (training.py)

- `TrainerConfig`는 JSON에서 변환된 값으로 생성.
- AMP는 `device='cuda'`이고 `dtype`이 `float16/bfloat16`일 때만 사용.
- 트레이딩 비용 반영: `buy/sell/sell_tax/slippage/hold_cost`만 반영(legacy `trans_cost` 제거).
- 체크포인트
  - 개선 시 저장(early_stopping 경로)
  - 개선이 한 번도 없으면 학습 종료 시 `checkpoint` 경로로 최종 체크포인트 저장(보완 로직 포함)

## 5. 실행 (main.py)

- `train_config.json`을 곧바로 로드해 `cfg` dict로 사용합니다.
- 장치 선택: `select_device(cfg.get("device"))` → mps → cuda → cpu 자동 선택.
- 파이프라인 실행: `DataPipeline.run(batch_size, num_workers, device)`
- 학습 실행: `TrainingLoop(model, train_loader, val_loader, config, loss_fn)`
- 선택적 저장
  - `history_out`: 학습 히스토리(JSON)
  - `model_out`: 학습 완료 후 최종 모델 state_dict
  - `config_out`: 실행에 사용된 최종 설정(JSON)

## 6. 사용 예시

```bash
python main.py  # 같은 폴더의 train_config.json 사용
```

- MPS/CPU 환경에서는 `dtype`을 `float32` 권장.
- CUDA 혼합정밀을 쓰려면 `device: "cuda"`, `dtype: "float16"|"bfloat16"` 설정.

## 7. 주요 차이점(간소화 리팩터)

- ScriptConfig 제거: 이제 모든 설정은 `train_config.json` 하나로 제어.
- FactorModel: `fit_all()`이 내부 `returns` 프로퍼티를 사용, 외부 인자 제거. `save()`에서 미학습 시 자동 학습.
- 거래 비용: `trans_cost` 제거, `buy/sell/sell_tax/slippage/hold_cost`만 유지.
- 체크포인트 보완: 개선이 없어도 마지막에 한 번 저장 보장.

## 8. dlsa-public 폴더와의 차이 확인

- 현재 저장소 내에 `dlsa-public` 폴더는 존재하지 않아 직접 diff 비교는 수행할 수 없습니다.
- 다만 일반적인 공개 버전과의 차이로 예상되는 점:
  - 설정 단일화(train_config.json), ScriptConfig 제거
  - MPS 자동 선택 포함(select_device), CUDA AMP 조건만 유지
  - 전처리 파이프라인의 자동 잔차 생성(ensure_residuals → save on-demand)
  - legacy 비용(trans_cost) 제거
  - 최종 모델/설정/체크포인트 저장 옵션 강화

`dlsa-public` 경로를 알려주시면 실제 코드 레벨로 diff 비교해 상세 보고 드리겠습니다.
