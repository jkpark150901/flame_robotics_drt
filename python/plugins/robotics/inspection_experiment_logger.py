from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np


class InspectionExperimentLogger:
    """검사 IK 수렴 과정을 파일로 저장하는 전용 로거.

    Viewer에서 직접 CSV/JSON 포맷을 만들지 않도록 분리한 I/O 모듈이다.
    로봇 해석, IK 계산, 시각화는 하지 않고 이미 계산된 trace와 메타데이터만 저장한다.
    """

    def __init__(self, root_dir, session_name: Optional[str] = None):
        """세션 단위 저장 폴더를 만든다.

        Args:
            root_dir: inspection IK 로그를 저장할 최상위 폴더.
            session_name: 세션 폴더명. None이면 현재 시각으로 만든다.

        Returns:
            없음.

        계산 과정:
            root/session_xxx 폴더를 생성하고 이후 save 호출은 success/fallback/failed/collision
            하위 폴더에 파일을 쌓는다.
        """
        root = Path(root_dir)
        if session_name is None:
            session_name = time.strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = root / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(value: Any) -> str:
        return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(value))

    @staticmethod
    def _status_label(ik_result: Optional[Dict[str, Any]]) -> str:
        result = ik_result or {}
        if bool(result.get("fallback", False)):
            status_label = "fallback"
        elif bool(result.get("success", False)):
            status_label = "success"
        else:
            status_label = "failed"
        if bool(result.get("collision", False)):
            status_label = f"{status_label}_collision"
        return status_label

    def save(
        self,
        *,
        robot_name: str,
        urdf_path: str,
        base_pose: Sequence[float],
        joint_names: Sequence[str],
        target_link_name: str,
        target_T: Optional[Any],
        goal_q: Sequence[float],
        trace: Sequence[Dict[str, Any]],
        ik_result: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        """IK trace와 메타데이터를 CSV/JSON으로 저장한다.

        Args:
            robot_name: 저장 파일명과 메타데이터에 기록할 로봇 이름.
            urdf_path: 재현용 URDF 경로.
            base_pose: 로봇 베이스 world pose.
            joint_names: q vector 각 항목의 joint 이름.
            target_link_name: IK target frame/link 이름.
            target_T: 목표 TCP world transform. None이면 JSON에 null로 저장한다.
            goal_q: IK 결과 q 또는 fallback q.
            trace: iteration별 q, error, TCP 위치 기록.
            ik_result: success/fallback/collision/solver 등 IK 요약 dict.

        Returns:
            {"csv": "...", "meta": "..."} 또는 저장할 trace가 없으면 None.

        계산 과정:
            1. 결과 상태와 normalize 여부로 파일명을 만든다.
            2. iteration trace를 wide CSV로 저장한다.
            3. 같은 stem의 JSON에 URDF, target transform, goal q, IK 요약을 저장한다.
        """
        if not trace:
            return None

        status_label = self._status_label(ik_result)
        status_dir = "collision" if "collision" in status_label else status_label
        out_dir = self.session_dir / status_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_robot = self._safe_name(robot_name)
        normalize_label = "normalized" if bool((ik_result or {}).get("normalize", False)) else "raw"
        stem = f"inspection_ik_{stamp}_{safe_robot}_{normalize_label}_{status_label}"
        csv_path = out_dir / f"{stem}.csv"
        json_path = out_dir / f"{stem}.json"

        joint_names = list(joint_names)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "iteration",
                "err_norm",
                "position_error",
                "orientation_error",
                "tcp_x",
                "tcp_y",
                "tcp_z",
            ] + [f"q{i}_{name}" for i, name in enumerate(joint_names)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in trace:
                q = np.asarray(row.get("q", []), dtype=float).reshape(-1)
                tcp = np.asarray(row.get("tcp_world", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
                data = {
                    "iteration": int(row.get("iteration", 0)),
                    "err_norm": float(row.get("err_norm", np.nan)),
                    "position_error": float(row.get("position_error", np.nan)),
                    "orientation_error": float(row.get("orientation_error", np.nan)),
                    "tcp_x": float(tcp[0]) if tcp.size > 0 else np.nan,
                    "tcp_y": float(tcp[1]) if tcp.size > 1 else np.nan,
                    "tcp_z": float(tcp[2]) if tcp.size > 2 else np.nan,
                }
                for i, name in enumerate(joint_names):
                    data[f"q{i}_{name}"] = float(q[i]) if i < q.size else np.nan
                writer.writerow(data)

        meta = {
            "robot_name": robot_name,
            "urdf_path": os.path.abspath(str(urdf_path or "")),
            "base_pose": list(base_pose),
            "joint_names": joint_names,
            "target_link_name": target_link_name,
            "csv_path": str(csv_path),
            "target_T": None if target_T is None else np.asarray(target_T, dtype=float).tolist(),
            "goal_q": np.asarray(goal_q, dtype=float).reshape(-1).tolist(),
            "ik_result": ik_result or {},
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return {"csv": str(csv_path), "meta": str(json_path)}
