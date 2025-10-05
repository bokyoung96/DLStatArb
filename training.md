# training.py 안내

## 전체 역할

`training.py`는 모델 학습을 총괄하는 모듈입니다. 데이터로더에서 `(panel, mask, next_residual)` 배치를 받아 손익(PnL)을 계산하고, 목적 함수에 맞는 손실을 산출한 뒤 최적화·평가·조기 종료·체크포인트를 모두 담당합니다. `main.py`가 준비한 설정(`TrainerConfig`)을 토대로 여기서 학습이 진행됩니다.

## 주요 구성 요소

- `TrainerConfig`: 학습률, 에폭 수, 거래 비용, 목적 함수(`sharpe`/`meanvar`/`sqrtMeanSharpe`), 가중치 정규화(`softmax` 또는 `l1`), 조기 종료, 체크포인트 경로 등 학습 관련 옵션을 모두 캡슐화합니다.
- 손실 모듈
  - `SharpeLoss`: 1차원 손익 벡터를 입력받아 샤프비율을 음수로 반환합니다.
  - `MeanVarianceLoss`: 연평균 수익률(×252)과 변동성(×15.9)의 조합으로 평균-분산 목적을 최소화합니다.
  - `SqrtMeanSharpeLoss`: 평균의 부호를 유지한 채 제곱근 스케일로 샤프비율을 최적화합니다.
- `_renorm_weights()`: 모델 출력(`weights`, `scores`)을 최종 매수비중으로 변환합니다. `softmax`는 모델이 생성한 확률 그대로, `l1`은 L1 정규화로 롱·숏 비중을 모두 허용합니다.
- `_apply_costs()`: 연속 시점 간 가중치 변화(L1)와 숏 포지션 절댓값을 이용해 거래비용(`trans_cost`), 보유비용(`hold_cost`)을 손익에서 차감합니다.
- AMP 유틸리티(`_should_use_amp`, `_make_grad_scaler`, `_autocast_context`): 설정된 dtype이 `float16` 또는 `bfloat16`이고 디바이스가 GPU일 때만 `torch.amp` 기반 혼합정밀과 `GradScaler`를 활성화합니다. 구버전 CUDA AMP API는 더 이상 사용하지 않습니다.

## 학습 루프 (`train_one_epoch`)

1. 배치 로딩: 데이터로더는 `(panel, mask, next_residual)` 튜플을 시간 순서대로 제공해야 합니다. 다른 순서를 섞으면 거래 비용 계산이 틀어집니다.
2. 디바이스 전송: `panel`, `mask`, `next_residual`을 `TrainerConfig.device`로 이동하고, AMP가 활성화됐다면 `torch.amp.autocast(device_type="cuda", dtype=config.dtype)` 컨텍스트를 사용합니다.
3. 모델 추론: `StatArbModel(panel, mask)`가 softmax 가중치와 점수를 반환합니다.
4. 가중치 정규화 & 손익 계산: `_renorm_weights`로 유효 자산만 남기고, 다음 시점 잔차(`next_residual`)와 곱해 PnL을 계산한 뒤 `_apply_costs`로 거래/보유 비용을 반영합니다.
5. 손실 산출: 주입된 `loss_fn(pnl)`이 목적 함수에 맞는 스칼라 손실을 반환합니다.
6. 역전파 & 최적화: 혼합정밀이면 `GradScaler.scale(loss).backward()` → `GradScaler.step()` → `GradScaler.update()`, 아니면 일반 `loss.backward()` 후 `optimizer.step()`을 수행합니다. 필요 시 `grad_clip`으로 기울기를 L2 노름 기준으로 잘라냅니다.

## 평가 루프 (`evaluate`)

- 모델을 `eval()` 모드로 두고 동일한 전처리를 수행하지만 파라미터를 업데이트하지 않습니다.
- 각 배치의 손실 값과 PnL을 축적해 평균 손실, 평균 수익률, 변동성, 샤프비율을 계산 후 딕셔너리로 반환합니다.

## TrainingLoop 동작 순서

1. 모델을 지정한 디바이스와 dtype으로 이동합니다.
2. 목적 함수에 따라 손실 모듈을 준비하고, 기본 `AdamW` 또는 사용자 정의 옵티마이저 팩토리를 생성합니다.
3. `config.dtype`이 `float16`/`bfloat16`이고 디바이스가 GPU면 AMP/GradScaler를 켜고, 아니면 일반 FP32 학습을 유지합니다.
4. 에폭마다 `train_one_epoch`를 호출해 학습하고, `eval_interval` 주기마다 `evaluate`로 검증 지표를 기록합니다.
5. `early_stopping=True`면 검증 손실이 개선되지 않았을 때 patience를 증가시키고, 한도를 넘으면 학습률을 `lr_decay`만큼 줄인 뒤 체크포인트를 복원합니다. `early_stopping_max_trials` 횟수를 초과하면 학습을 중단합니다.
6. 각 에폭의 결과(`epoch`, `train_loss`, `val_loss`, `val_sharpe` 등)를 리스트에 쌓아 최종 반환합니다.

## 실행 및 설정 가이드

1. `main.py`를 실행하면 기본 설정(논문과 동일한 `lookback=30`, `batch_size=200`, `weight_scheme="l1"`, 거래비용 5bp/10bp)이 적용됩니다.
2. 설정을 바꾸고 싶다면 프로젝트 루트의 `train_config.json`을 수정하세요. 예를 들어 Mac의 MPS를 사용하려면 `"device": "mps"`, `"dtype": "float32"`를 지정하면 됩니다. 존재하는 키만 덮어쓰며, 나머지는 `ScriptConfig` 기본값을 사용합니다.
3. JSON 예시:
   ```json
   {
     "epochs": 120,
     "weight_scheme": "l1",
     "trans_cost": 0.0005,
     "hold_cost": 0.0001,
     "log_test": true,
     "checkpoint": "checkpoints/stat_arb.pt"
   }
   ```
4. 학습 기록을 남기고 싶다면 `history_out`에 JSON 저장 경로를, 중간 저장을 원하면 `checkpoint` 경로를 지정하세요.
5. AMP를 사용하려면 GPU 환경에서 `dtype`을 `"float16"` 또는 `"bfloat16"`으로 설정하세요. CPU에서는 자동으로 FP32 경로로 돌아갑니다.

## 추가 팁

- `weight_scheme="l1"`는 롱·숏 포트폴리오를 허용하므로 논문 결과 재현에 적합합니다. `softmax`는 롱온리 전략에 사용하세요.
- 거래 비용을 끄고 싶을 땐 `trans_cost=0`, `hold_cost=0`으로 설정하면 됩니다.
- 테스트 세트 성능을 로그로 확인하려면 `log_test`를 `true`로 두거나, 학습 후 `training.evaluate`를 직접 호출하세요.
- 커스텀 손실을 쓰고 싶다면 `TrainingLoop(..., loss_fn=my_loss_module)`로 주입하면 손익 벡터에 바로 적용됩니다.
