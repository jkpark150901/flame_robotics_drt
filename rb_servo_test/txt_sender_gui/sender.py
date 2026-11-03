"""TXT recipe reader and TCP sender. No GUI dependencies."""
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import re
import select
import socket
import threading
import time

ACK = b'The command was executed'


@dataclass(frozen=True)
class Step:
    line: int
    command: str = ''
    delay: float = 0.
    period: float = 0.


@dataclass(frozen=True)
class Recipe:
    steps: tuple
    loop_start: int

    @property
    def periods(self):
        return sorted({s.period for s in self.steps if s.period})


def parse_recipe(text):
    steps, loop_start = [], None
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    for line_no, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            if stripped.startswith(('# LOOP BODY STARTS', '# LOOP FOREVER FROM HERE')):
                if loop_start is not None:
                    raise ValueError(f'{line_no}행: LOOP 시작 표식이 중복되었습니다.')
                loop_start = len(steps)
            wait = re.match(rf'#\s*WAIT\s+({number})\s*s\b', stripped, re.I)
            if wait:
                delay = float(wait[1])
                if not math.isfinite(delay) or delay < 0:
                    raise ValueError(f'{line_no}행: 대기 시간이 잘못되었습니다.')
                steps.append(Step(line_no, delay=delay))
            continue
        wait = re.fullmatch(rf'wait\s*\(\s*({number})\s*sec\s*\)', stripped, re.I)
        if wait:
            if not math.isfinite(float(wait[1])) or float(wait[1]) < 0:
                raise ValueError(f'{line_no}행: 대기는 음수일 수 없습니다.')
            steps.append(Step(line_no, delay=float(wait[1])))
            continue
        # Preserve command text; only remove an optional trailing comment.
        cmd = raw.split('#', 1)[0].rstrip()
        match = re.fullmatch(r'\s*(move_j|move_servo_j|move_finish_wait|move_jb2_clear|move_jb2_add|move_jb2_run)\((.*)\)', cmd)
        if not match:
            raise ValueError(f'{line_no}행: 지원하지 않는 전문입니다: {stripped}')
        kind, body = match.groups()
        period = 0.
        if kind in ('move_j', 'move_servo_j', 'move_jb2_add'):
            values = re.fullmatch(r'\s*jnt\[([^\]]+)\]\s*,(.*)', body)
            if not values:
                raise ValueError(f'{line_no}행: jnt[...] 형식을 확인하세요.')
            try:
                joints = [float(v) for v in values[1].split(',')]
                params = [float(v) for v in values[2].split(',')]
            except ValueError:
                raise ValueError(f'{line_no}행: 숫자 형식 오류') from None
            if len(joints) != 6 or not all(math.isfinite(v) for v in joints+params):
                raise ValueError(f'{line_no}행: 유한한 6개 관절각과 파라미터가 필요합니다.')
            if kind == 'move_servo_j':
                if len(params) != 4 or not .002 <= params[0] <= .1:
                    raise ValueError(f'{line_no}행: Servo J의 t1 주기는 0.002~0.1초여야 합니다.')
                period = params[0]
            elif kind == 'move_j' and (len(params) not in (2,3) or min(params[:2]) <= 0):
                raise ValueError(f'{line_no}행: Move J 파라미터 오류')
            elif kind == 'move_jb2_add' and (len(params) != 4 or min(params[:2]) <= 0 or not 0 <= params[-1] <= 100):
                raise ValueError(f'{line_no}행: JB2 파라미터 오류')
        elif kind in ('move_finish_wait','move_jb2_clear') and body.strip():
            raise ValueError(f'{line_no}행: 인자가 없어야 합니다.')
        elif kind == 'move_jb2_run' and body.strip() not in ('','0','1'):
            raise ValueError(f'{line_no}행: run 인자는 0 또는 1입니다.')
        steps.append(Step(line_no, cmd, period=period))
    if not any(s.command for s in steps):
        raise ValueError('전송할 명령이 없습니다.')
    if loop_start is not None and loop_start >= len(steps):
        raise ValueError('LOOP 본문이 비어 있습니다.')
    return Recipe(tuple(steps), loop_start if loop_start is not None else 0)


class Sender:
    def __init__(self, emit=None, cancel=None, timeout=120., ending='LF', max_lag_ms=10.):
        self.emit = emit or (lambda kind, data: None)
        self.cancel = cancel or threading.Event()
        self.timeout = timeout
        self.ending = {'LF':b'\n', 'CRLF':b'\r\n'}[ending]
        self.max_lag = max_lag_ms/1000
        self.pending = deque()
        self.buffer = b''
        self.records = []
        self.sock = None

    def check(self):
        if self.cancel.is_set():
            raise InterruptedError('후속 전송을 중단했습니다. 이미 실행 중인 로봇 모션은 자동 정지되지 않습니다.')
        if self.pending and time.perf_counter()-self.pending[0] > self.timeout:
            raise TimeoutError('명령 완료 응답 시간 초과. 후속 전송을 중단합니다.')

    def receive(self, timeout=0.):
        self.check()
        if not select.select([self.sock], [], [], max(0.,timeout))[0]:
            return
        data = self.sock.recv(65536)
        if not data:
            raise EOFError('서버가 연결을 종료했습니다.')
        self.emit('RX', data.decode('utf-8',errors='replace'))
        self.buffer += data
        lower = self.buffer.lower()
        if b'not allowed' in lower or b'error' in lower:
            raise ValueError('서버 오류/거부 응답. RX 로그를 확인하세요.')
        # Handles ACK tokens split across recv calls or coalesced in a chunk.
        while ACK in self.buffer:
            _, self.buffer = self.buffer.split(ACK,1)
            if self.pending:
                self.pending.popleft()
        self.buffer = self.buffer[-4096:]

    def wait_until(self, deadline):
        while True:
            self.check()
            remaining = deadline-time.perf_counter()
            if remaining <= 0:
                self.receive(0.)
                return
            self.receive(min(remaining,.005))

    def barrier(self):
        while self.pending:
            self.receive(.005)
        self.check()

    def send(self, step, cycle, planned=None):
        self.check()
        started = time.perf_counter()
        self.pending.append(started)
        self.sock.sendall(step.command.encode('utf-8')+self.ending)
        actual = time.perf_counter()
        previous = self.records[-1] if self.records else None
        self.records.append(dict(cycle=cycle,line=step.line,servo_period_s=step.period,
            planned_s=planned,actual_s=actual,late_ms=(actual-planned)*1000 if planned is not None else '',
            actual_interval_ms=(actual-previous['actual_s'])*1000
                if step.period and previous and previous['servo_period_s'] else '',
            command=step.command))
        if planned is not None and actual-planned > min(self.max_lag,step.period/2):
            raise TimeoutError('Servo 소켓 전송이 허용 지연을 초과했습니다. 후속 전송을 중단합니다.')
        self.emit('TX', f'{step.line}행 {step.command}')

    def execute(self, recipe, repeats):
        cycle = 0
        while repeats == 0 or cycle < repeats:
            cycle += 1
            steps = recipe.steps if cycle == 1 else recipe.steps[recipe.loop_start:]
            deadline = None
            self.emit('STATUS', f'{cycle}회 실행 중')
            for step in steps:
                if step.period:
                    if deadline is None:
                        # Drain ALL previous responses, including setup + completion wait.
                        self.barrier()
                        deadline = time.perf_counter()
                    self.wait_until(deadline)
                    if time.perf_counter()-deadline > min(self.max_lag,step.period/2):
                        raise TimeoutError('Servo 주기 지연 초과: 밀린 지령을 몰아서 보내지 않고 중단합니다.')
                    self.send(step,cycle,deadline)
                    deadline += step.period
                else:
                    if deadline is not None:
                        # Last target has one full t1 before sending finish_wait.
                        self.wait_until(deadline)
                        deadline = None
                    self.barrier()
                    if step.command:
                        self.send(step,cycle)
                        self.barrier()
                    else:
                        self.emit('STATUS', f'{step.line}행: {step.delay:g}초 대기')
                        self.wait_until(time.perf_counter()+step.delay)
            if deadline is not None:
                self.wait_until(deadline)
            self.barrier()
        self.emit('STATUS', '전송 및 명령 응답 대기 완료')

    def run(self, host, port, recipe, repeats=1):
        try:
            with socket.create_connection((host,port),timeout=3.) as sock:
                self.sock = sock
                sock.settimeout(.1)
                sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                self.emit('STATUS',f'연결됨: {host}:{port}')
                self.execute(recipe,repeats)
        finally:
            self.sock = None
