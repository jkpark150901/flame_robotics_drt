"""Collect the evaluation summaries of every result folder under runs/ into one CSV (runs/summary.csv).

Each runs/<algorithm>_<optimizer>_<scenario>/eval/summary.json (written by evaluate_path_plans.py --save-runs or
evaluate_plan_run.py) becomes one row: the folder name split into algorithm / optimizer / scenario, then the
planner-level metrics (see eval/METRICS.md for the definitions).

    python python/runs_summary.py [--runs-dir runs] [--out runs/summary.csv]
"""
import argparse
import csv
import json
import pathlib

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent

COLUMNS = [
    # (csv column, summary.json key, scale, digits)
    ("n_requests", "n_requests", 1, 0),
    ("n_planned", "n_planned", 1, 0),
    ("n_safe", "n_safe", 1, 0),
    ("planned_rate_pct", "planned_rate_pct", 1, 1),
    ("SR_safe_pct", "SR_safe_pct", 1, 1),
    ("C_eff_mean_safe_mm", "C_eff_mean_safe_mm", 1, 1),
    ("C_eff_mean_planned_mm", "C_eff_mean_planned_mm", 1, 1),
    ("C_eff_min_planned_mm", "C_eff_min_planned_mm", 1, 1),
    ("C_nom_min_planned_mm", "C_nom_min_planned_mm", 1, 1),
    ("E_d_mean_mm", "E_d_mean_mm", 1, 2),
    ("E_theta_mean_deg", "E_theta_mean_deg", 1, 1),
    ("approach_dir_error_mean_deg", "approach_dir_error_mean_deg", 1, 1),
    ("L_ee_mean_m", "L_ee_mean_m", 1, 3),
    ("L_ee_total_m", "L_ee_total_m", 1, 3),
    ("T_motion_mean_s", "T_motion_mean_s", 1, 2),
    ("T_motion_total_s", "T_motion_total_s", 1, 2),
    ("L_q_normalized_mean", "L_q_normalized_mean", 1, 3),
    ("T_plan_mean_s", "T_plan_mean_s", 1, 2),
    ("T_plan_p50_s", "T_plan_p50_s", 1, 2),
    ("T_plan_p95_s", "T_plan_p95_s", 1, 2),
    ("timeout_rate_pct", "timeout_rate_pct", 1, 1),
]


# summary.csv column -> (지표, 단위, 설명)
DESCRIPTIONS = {
    "run": ("결과 폴더", "-", "runs/ 아래 폴더 이름 {알고리즘}_{최적화}_{시나리오}"),
    "algorithm": ("경로 계획 알고리즘", "-", "OMPL RRTConnect / RRTstar / InformedRRTstar"),
    "optimizer": ("경로 최적화", "-", "none = 플래너 결과 그대로, stomp / gpmp2 / trajopt = APF 장애물 비용 기반 최적화"),
    "scenario": ("시나리오", "-", "p1 = sample/planning_snapshot_p1_2.pkl"),
    "n_requests": ("요청 수", "건", "평가한 목표 수 (실패 포함)"),
    "n_planned": ("계획 성공 수", "건", "경로가 만들어진 목표 수"),
    "n_safe": ("안전 성공 수", "건", "계획 성공 & C_eff >= 0 & 필수 비관통 & E_d <= eps_d & E_theta <= eps_theta"),
    "planned_rate_pct": ("계획 성공률", "%", "n_planned / n_requests"),
    "SR_safe_pct": ("안전 성공률 SR_safe", "%", "n_safe / n_requests"),
    "C_eff_mean_safe_mm": ("유효 여유거리 평균 (안전 성공)", "mm", "C_eff = 경로 전체 로봇-장애물 최소 거리 - 안전 마진"),
    "C_eff_mean_planned_mm": ("유효 여유거리 평균 (계획 성공)", "mm", ""),
    "C_eff_min_planned_mm": ("유효 여유거리 최소 (계획 성공)", "mm", "음수 = 안전 마진 침범 (더 작으면 관통)"),
    "C_nom_min_planned_mm": ("명목 여유거리 최소 (계획 성공)", "mm", "마진 빼기 전 최소 거리, 음수 = 관통"),
    "E_d_mean_mm": ("검사 거리 오차 평균", "mm", "|최종 센서-배관 거리 - d*| (DDA)"),
    "E_theta_mean_deg": ("검사 방향 오차 평균", "deg", "최종 센서 축과 배관 표면 법선 사이 각도 (DDA, 기준 5°)"),
    "approach_dir_error_mean_deg": ("접근 방향 오차 평균", "deg", "마지막 0.1 m EE 이동 방향과 표면 법선 사이 각도"),
    "L_ee_mean_m": ("궤적 길이 평균 (EE)", "m", "계획 성공 경로의 TCP 경로 길이 평균"),
    "L_ee_total_m": ("궤적 길이 합 (EE)", "m", "계획 성공 경로의 TCP 경로 길이 합"),
    "T_motion_mean_s": ("모션 수행 시간 평균", "s", "사다리꼴 속도 프로파일 (회전 50 °/s, 30 °/s², 구간별 최대 변위 관절 기준 / "
                                             "직동 0.5 m/s, 0.3 m/s²) - 플레이백 시간과 같음"),
    "T_motion_total_s": ("모션 수행 시간 합", "s", "계획 성공 경로의 모션 시간 합 (성공 수가 다르면 직접 비교 주의)"),
    "L_q_normalized_mean": ("정규화 관절 경로 길이 평균", "-", "관절 범위로 나눈 관절 공간 경로 길이"),
    "T_plan_mean_s": ("계획 시간 평균", "s", "목표 1건 계획 소요 시간 (IK + 경로 계획 + 최적화 + 검증), 실패 포함"),
    "T_plan_p50_s": ("계획 시간 중앙값", "s", ""),
    "T_plan_p95_s": ("계획 시간 95% 분위", "s", ""),
    "timeout_rate_pct": ("타임아웃 비율", "%", "timeout(상태검사 상한 또는 시간 제한)으로 끝난 목표 비율"),
    "waypoints_mean": ("웨이포인트 수 평균", "개", "계획 성공 경로의 웨이포인트 수 (경로 계획/최적화 결과 q_path 점 수)"),
    "waypoints_min": ("웨이포인트 수 최소", "개", ""),
    "waypoints_max": ("웨이포인트 수 최대", "개", ""),
    "waypoints_total": ("웨이포인트 수 합", "개", ""),
    "eval_samples_mean": ("평가 샘플 수 평균", "개", "여유거리/충돌 평가 점 수 - 웨이포인트 사이를 관절 공간 0.02 (rad/m 노름) "
                                             "간격으로 보간, 경로당 최대 250 (넘으면 균등 추출)"),
    "n_paths_sample_capped": ("샘플 상한에 걸린 경로 수", "개", "평가 샘플이 250 상한에 걸려 실제 간격이 0.02보다 넓어진 경로 수"),
    "SR_safe_ci95_low_pct": ("안전 성공률 95% 신뢰구간 하한", "%", "Wilson 구간"),
    "SR_safe_ci95_high_pct": ("안전 성공률 95% 신뢰구간 상한", "%", "Wilson 구간"),
}


def write_metrics_md(runs_dir, fields, first_eval_dir=None):
    """runs/METRICS.md: the summary.csv columns, then the full metric definitions of the evaluation (eval/METRICS.md)."""
    lines = ["# 평가 지표 정의", "", "## summary.csv 컬럼", "", "| 컬럼 | 지표 | 단위 | 설명 |", "|---|---|---|---|"]
    for col in fields:
        name, unit, desc = DESCRIPTIONS.get(col, (col, "", ""))
        lines.append(f"| `{col}` | {name} | {unit} | {desc.replace('|', chr(92) + '|')} |")
    if first_eval_dir is not None and (first_eval_dir / "METRICS.md").exists():
        body = (first_eval_dir / "METRICS.md").read_text(encoding="utf-8").splitlines()
        body = [l.replace("# 평가 지표 (path_plan_metric.md)", "## 평가 지표 상세 (각 폴더 eval/targets.csv, eval/summary.json)")
                .replace("## 목표별", "### 목표별").replace("## 전체", "### 전체") for l in body]
        lines += [""] + body
    (pathlib.Path(runs_dir) / "METRICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value, scale, digits):
    if value is None:
        return ""
    try:
        v = float(value) * scale
    except (TypeError, ValueError):
        return value
    return f"{v:.{digits}f}" if digits else str(int(round(v)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(ROOT_PATH / "runs"))
    ap.add_argument("--out", default=None, help="default: <runs-dir>/summary.csv")
    args = ap.parse_args()
    runs_dir = pathlib.Path(args.runs_dir)
    out = pathlib.Path(args.out) if args.out else runs_dir / "summary.csv"

    rows = []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        path = run / "eval" / "summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = next(iter((data.get("summaries") or {}).values()), {})
        parts = run.name.split("_")
        algorithm, optimizer, scenario = (parts + ["", "", ""])[:3] if len(parts) >= 3 else (run.name, "", "")
        if len(parts) > 3:   # scenario names may contain "_"
            scenario = "_".join(parts[2:])
        ci = summary.get("SR_safe_ci95_pct") or [None, None]
        # waypoints of the planned paths (plan-run summary.csv) and evaluation samples (eval/targets.csv)
        wps, samples, capped = [], [], 0
        if (run / "summary.csv").exists():
            for r in csv.DictReader(open(run / "summary.csv", newline="", encoding="utf-8")):
                if r.get("status") == "success" and r.get("n_waypoints"):
                    wps.append(int(r["n_waypoints"]))
        if (run / "eval" / "targets.csv").exists():
            for r in csv.DictReader(open(run / "eval" / "targets.csv", newline="", encoding="utf-8")):
                if r.get("planned") == "True" and r.get("n_dense_samples") not in (None, "", "None"):
                    n = int(float(r["n_dense_samples"]))
                    samples.append(n)
                    interp = r.get("n_interp_samples")
                    if interp not in (None, "", "None"):
                        capped += int(float(interp) > n)
                    else:   # evaluated before n_interp_samples existed: at the cap = capped
                        capped += int(n >= 250)
        row = {"run": run.name, "algorithm": algorithm, "optimizer": optimizer, "scenario": scenario}
        row.update({col: _fmt(summary.get(key), scale, digits) for col, key, scale, digits in COLUMNS})
        row["waypoints_mean"] = _fmt(sum(wps) / len(wps), 1, 1) if wps else ""
        row["waypoints_min"] = min(wps) if wps else ""
        row["waypoints_max"] = max(wps) if wps else ""
        row["waypoints_total"] = sum(wps) if wps else ""
        row["eval_samples_mean"] = _fmt(sum(samples) / len(samples), 1, 1) if samples else ""
        row["n_paths_sample_capped"] = capped if samples else ""
        row["SR_safe_ci95_low_pct"] = _fmt(ci[0], 1, 1)
        row["SR_safe_ci95_high_pct"] = _fmt(ci[1], 1, 1)
        rows.append(row)

    fields = (["run", "algorithm", "optimizer", "scenario"] + [c for c, *_ in COLUMNS]
              + ["waypoints_mean", "waypoints_min", "waypoints_max", "waypoints_total", "eval_samples_mean",
                 "n_paths_sample_capped", "SR_safe_ci95_low_pct", "SR_safe_ci95_high_pct"])
    with open(out, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    first_eval = next((p / "eval" for p in sorted(runs_dir.iterdir()) if (p / "eval" / "METRICS.md").exists()), None)
    write_metrics_md(runs_dir, fields, first_eval)
    print(f"{len(rows)} run(s) -> {out}, metric definitions -> {runs_dir / 'METRICS.md'}")


if __name__ == "__main__":
    main()
