# 협동로봇 검사 경로계획 평가 지표 (1차안)

## 1. 평가 목적 및 전제

- **목적:** 경로 추종 오차 및 센서·정합 오차를 고려하여, **검사 목표 자세를 만족**하면서 **비의도적 충돌에 대한 안전 이격거리를 충분히 확보**하고 **짧은 시간에 경로를 계획**하는 알고리즘 평가.
- **대상:** 협동로봇 전체 링크, 검사 툴, 필요 시 레일·갠트리 외부축. 관절 상태는 외부축을 포함한 \(q\in\mathbb R^n\).
- **핵심 예외:** 검사 센서–검사 대상의 **의도된 근접**은 일반적인 최대 이격거리 평가에서 제외하되, **비관통·최소 물리 간격 및 목표 거리·방향**은 별도 검증한다.
- 모든 경로 샘플뿐 아니라 **웨이포인트 사이의 연속 경로**도 충돌 검사한다(CCD 또는 충분히 보수적인 보간·swept volume 검증).

## 2. 통합 평가 정의

경로 \(Q=\{q(s):s\in[0,1]\}\)에 대해, 링크 \(l\)와 장애물 \(o\)의 signed distance를 \(d_{l,o}(q(s))\)라 한다(비충돌 양수, 접촉 0, 침투 음수). 충돌 검사 쌍을 다음과 같이 구분한다.

- \(\mathcal P_{\rm safe}(s)\): **안전 이격거리 최적화 대상** 링크–장애물 쌍. 검사 센서–대상 쌍은 근접 접근 구간에서 제외하며, 나머지 로봇 링크–검사 대상 쌍은 계속 포함한다.
- \(\mathcal P_{\rm mandatory}\): **항상 비관통을 확인할 대상** 쌍(검사 센서–검사 대상 포함).
- \(m_{l,o}(s)\): 추종·환경 인식/정합 오차를 작업공간 거리 단위로 환산한 **쌍별 안전 마진**. 불확실성을 확률적 방식으로 표현한다면 그 신뢰수준을 함께 명시한다.

### 2.1 핵심 지표: 유효 최소 이격거리

\[
\boxed{C_{\rm eff}(Q)=\min_{s\in[0,1]}\ \min_{(l,o)\in\mathcal P_{\rm safe}(s)}\big[d_{l,o}(q(s))-m_{l,o}(s)\big]}
\]

- **클수록 좋음.** 평가 대상의 정의가 바뀌면 값의 의미도 바뀌므로, 모든 방법에 동일한 구간·충돌 쌍 정책을 적용한다.
- 검사 대상에 가까워지는 **센서 자체가 전체 최소거리 값을 강제로 결정하지 않도록** 한다.
- 검사 센서–대상 쌍은 제외하더라도 \(d_{\rm tool,target}(q(s))\ge d_{\rm physical,min}\) 및 비관통 여부를 **전체 경로에 대해** 검증한다.

### 2.2 검사 정확도: 목표 거리 및 방향

검사 종료 상태 \(q(1)\)에서 다음 두 오차를 계산한다.

\[
\boxed{E_d=\left|d_{\rm tool,target}(q(1))-d^*\right|}
\]
\[
\boxed{E_\theta=\cos^{-1}\!\left(\operatorname{clip}(\mathbf a_{\rm sensor}^{\top}(-\mathbf n_{\rm target}),-1,1)\right)}
\]

- \(d^*\): 검사 센서의 목표 이격거리, \(\mathbf a_{\rm sensor}\): 센서가 바라보는 단위 방향, \(\mathbf n_{\rm target}\): 검사 표면의 외향 단위 법선.
- 판정 조건: \(E_d\le\epsilon_d\), \(E_\theta\le\epsilon_\theta\). 센서의 목표 자세가 완전히 주어졌다면 위치·자세 오차도 추가 기록한다.
- 검사 접근 방향이 중요하면 마지막 접근 구간의 진행방향과 \(-\mathbf n_{\rm target}\)의 정렬 오차를 보조 지표로 사용한다.

### 2.3 강건성: 추종·센서 오차에 따른 충돌/위반율

오차가 반영된 실제 실행 궤적 \(q_{\rm exec}^{(k)}(s)\)와 실제 장애물 \(O^{(k)}\)에 대해 \(k=1,\ldots,N\)회 검증한다.

\[
\boxed{\widehat P_{\rm collision}=\frac{1}{N}\sum_{k=1}^N \mathbf1\!\left[\min_{s,(l,o)\in\mathcal P_{\rm mandatory}}d_{l,o}^{(k)}(s)\le0\right]}
\]
\[
\boxed{\widehat P_{\rm violation}=\frac{1}{N}\sum_{k=1}^N \mathbf1\!\left[C_{\rm eff}^{(k)}<0\ \text{또는 검사 제약 위반}\right]}
\]

- **작을수록 좋음.** 안전 마진이 이미 불확실성 경계를 포함한다면 Monte Carlo 평가에서 이를 중복 적용하지 않도록 **실제 거리 위반 판정 기준**을 명시한다.
- 경로 생성 시의 센서 노이즈(인식·계획)와 실행 중 추종 오차를 구분한다. 실제 노이즈에 대한 강건한 **계획 능력**까지 평가하려면 노이즈 관측으로 경로를 새로 생성해야 한다.

### 2.4 계획 효율 및 유효 계획 성공률

\[
\boxed{T_{\rm plan}=t_{\rm validated\ path}-t_{\rm request}}
\]

- 전처리·탐색/추론·최적화·후처리·최종 검증을 포함. **평균, 중앙값(P50), P95, timeout 비율** 보고. RL 오프라인 학습시간은 별도로 기록한다.

\[
\boxed{SR_{\rm safe}=\frac{\#\{\text{기한 내 경로 생성, 검사 제약 만족, 충돌/강건성 기준 만족}\}}{\#\{\text{전체 계획 요청}\}}\times100\%}
\]

- 성공한 경로의 이격거리 통계만 제시하면 실패 사례가 누락되므로, **성공률과 이격거리를 함께** 보고한다.

## 3. 보조 지표

| 지표 | 정의 | 방향/참고 |
|---|---|---|
| Nominal Minimum Clearance | \(C_{\rm nom}=\min_{s,(l,o)\in\mathcal P_{\rm safe}(s)}d_{l,o}(q(s))\) | ↑, 안전 마진 적용 전 |
| Joint-space Path Length | \(L_q=\sum_{i=1}^{M-1}\sqrt{(q_{i+1}-q_i)^\top W(q_{i+1}-q_i)}\) | ↓, 레일(m)·회전(rad) 단위 정규화 필요 |
| Executed Clearance | 실제 관절 엔코더 및 실제/검증 환경 기준 최소거리 | ↑, 시간순 데이터 사용 |
| DTW-RMSE | \(\sqrt{\frac1K\sum_{k=1}^{K}\|p_{i_k}-a_{j_k}\|_2^2}\) | ↓, EE 경로의 공간적 추종 오차; 링크 안전성 대용 불가 |
| 접근 방향 오차 | 최종 접근 경로 접선과 표면 법선 사이 각도 | ↓, 검사 품질이 접근 방향에 민감한 경우 |

## 4. 평가 의사코드

```python
for scene in test_scenes:                # 학습에 쓰지 않은 장면 포함
    for planner in planners:             # RRT, RRT+APF/STOMP, SAC, SAC+Planner 등
        for seed in seeds:
            obs = sample_noisy_observation(scene, seed)
            t0 = monotonic_time()
            Q = planner.plan(obs, deadline=T_max)
            Q = postprocess(Q) if Q is not None else None
            valid_geometry = continuous_collision_check(Q, scene.estimated_geometry)
            t_plan = monotonic_time() - t0

            if Q is None or not valid_geometry or t_plan > T_max:
                log_failure(planner, scene, seed, t_plan)
                continue

            C_eff = min_effective_clearance(Q, scene, safe_pairs_policy)
            E_d, E_theta = inspection_goal_errors(Q[-1], scene.target)
            intended_pair_ok = check_tool_target_nonpenetration(Q, scene)

            trial_clearances = []
            trial_collisions = []
            for k in range(N_uncertainty_trials):
                true_env = sample_true_environment_conditioned_on(obs, scene, k)
                executed = execute_or_simulate_tracking(Q, k)
                trial_clearances.append(actual_minimum_clearance(executed, true_env))
                trial_collisions.append(any_mandatory_collision(executed, true_env))

            safe = (C_eff >= 0 and intended_pair_ok
                    and E_d <= eps_d and E_theta <= eps_theta
                    and meets_robustness_requirement(trial_collisions))
            log_result(planner, scene, seed, t_plan, safe, C_eff,
                       E_d, E_theta, trial_clearances, trial_collisions)

report_success_rate_and_confidence_interval()
report_planning_time_P50_P95_and_timeout_rate()
report_clearance_by_successful_path_and_failures()
report_collision_and_violation_rate()
```

## 5. 핵심 비교 실험

1. **동일 시간 제한** \(T_{\max}\)에서 각 알고리즘의 \(SR_{\rm safe}\), \(C_{\rm eff}\), 충돌률 비교.
2. **동일 목표 유효 이격거리** \(C_{\rm target}\)에 최초 도달하는 시간 비교(미도달은 실패/검열로 보고).
3. **노이즈 수준별 강건성**: 추종 오차와 측정·정합 오차 크기를 단계적으로 증가시키며 성능 변화 분석.
4. **RL ablation**: 동일한 후처리·충돌검사를 둔 `기존 Planner`와 `RL 초기화 + 기존 Planner` 비교.
5. **검사 목표 고정**: 동일 검사 목표 거리·방향과 동일 로봇/레일 관절 제한을 모든 알고리즘에 적용.

### 결과표 양식

| Planner | Safe Success Rate ↑ (%) | Effective Clearance ↑ (mm) | Collision Rate ↓ (%) | 검사 거리 오차 ↓ (mm) | 검사 방향 오차 ↓ (°) | Planning P95 ↓ (s) |
|---|---:|---:|---:|---:|---:|---:|
| RRT-Connect | — | — | — | — | — | — |
| RRT + APF | — | — | — | — | — | — |
| RRT + STOMP | — | — | — | — | — | — |
| SAC | — | — | — | — | — | — |
| SAC + Planner | — | — | — | — | — | — |

> **해석 원칙:** 본 연구의 핵심은 `전체 경로에서 무조건 가장 먼 경로`가 아니라 **검사에 필요한 의도적 근접을 보장하면서, 비의도적 충돌 위험에 대한 유효 이격거리를 확보하고 계획시간을 줄이는 것**이다.
