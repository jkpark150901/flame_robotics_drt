"""Per-segment synchronized trapezoidal joint profiles, degrees and seconds.

Each waypoint is reached at zero velocity, including direction reversals.
Scaling each segment to whole transmission periods preserves its limits.
"""
import math


class ServoProfile:
    def __init__(self, waypoints, speed, accel, period):
        self.points = [tuple(float(v) for v in point) for point in waypoints]
        self.speed, self.accel, self.period = map(float, (speed, accel, period))
        if len(self.points) < 2 or any(len(p) != 6 for p in self.points):
            raise ValueError("6개 관절각을 갖는 웨이포인트가 2개 이상 필요합니다.")
        if not all(math.isfinite(v) for p in self.points for v in p):
            raise ValueError("관절각은 유한한 숫자여야 합니다.")
        if not all(math.isfinite(v) and v > 0 for v in (self.speed, self.accel, self.period)):
            raise ValueError("속도·가속도·주기는 유한한 양수여야 합니다.")
        if not 0.002 <= self.period <= 0.1:
            raise ValueError("전송 주기는 0.002~0.1초 범위로 입력하세요.")
        self.segments = []
        for start, end in zip(self.points, self.points[1:]):
            displacements = [abs(b - a) for a, b in zip(start, end)]
            distance = max(displacements)
            if distance == 0:
                continue
            velocity = self.speed
            acceleration = self.accel
            peak = min(velocity, math.sqrt(distance) * math.sqrt(acceleration))
            ramp = peak / acceleration
            duration = distance / peak + ramp
            if not math.isfinite(duration) or duration / self.period > 1_000_000:
                raise ValueError("구간당 전송 지령이 너무 많습니다. 속도·가속도·주기를 조정하세요.")
            count = math.ceil(duration / self.period)
            self.segments.append((start, end, distance, peak, ramp, duration, acceleration, count))
        if not self.segments:
            raise ValueError("서로 다른 웨이포인트를 추가하세요.")
        self.count = sum(segment[-1] for segment in self.segments)
        self.duration = self.count * self.period

    def samples(self):
        """Yield (send time, target); target is reached one period later."""
        index = 0
        for start, end, distance, peak, ramp, duration, acceleration, count in self.segments:
            for step in range(1, count + 1):
                t = duration * step / count
                if t < ramp:
                    traveled = 0.5 * acceleration * t * t
                elif t <= duration - ramp:
                    traveled = 0.5 * peak * ramp + peak * (t - ramp)
                else:
                    traveled = distance - 0.5 * acceleration * (duration - t) ** 2
                fraction = min(1.0, max(0.0, traveled / distance))
                target = end if step == count else tuple(a + (b - a) * fraction for a, b in zip(start, end))
                yield index * self.period, target
                index += 1
