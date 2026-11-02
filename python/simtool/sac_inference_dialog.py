"""Non-blocking saved-model inference dialog. The worker only produces simulation paths."""
import codecs
from datetime import datetime
import json
from pathlib import Path
import re
import sys

from PyQt6.QtCore import QProcess, QProcessEnvironment, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
                             QPushButton, QFileDialog, QCheckBox, QComboBox, QPlainTextEdit,
                             QProgressBar, QMessageBox, QWidget)

ROOT = Path(__file__).resolve().parents[2]


class SACInferenceDialog(QDialog):
    load_result = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('SAC Inference — 학습 모델 추론')
        self.resize(780, 600)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.result = None
        self.cancelled = False
        self.pending = ''
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        layout = QVBoxLayout(self)
        note = QLabel('저장된 스냅샷과 기준 경로를 학습된 SAC로 보정합니다.\n'
                      '학습에 사용한 로봇만 추론하며, 결과는 Start Simulation으로 재생합니다.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.fields = QWidget()
        form = QFormLayout(self.fields)
        self.edits = {}
        for key, label in [('model', '학습 모델 (.zip)'), ('config', '학습 설정 (.json)'),
                           ('snapshot', '스냅샷 (.pkl)'), ('reference', '기준 경로 폴더')]:
            row = QHBoxLayout()
            edit = QLineEdit()
            button = QPushButton('찾기…')
            button.clicked.connect(lambda checked=False, k=key: self._browse(k))
            row.addWidget(edit)
            row.addWidget(button)
            form.addRow(label, row)
            self.edits[key] = edit
        self.target = QLineEdit()
        self.target.setPlaceholderText('비우면 전체 / 예: Point 1 - Inspection pose 1')
        form.addRow('목표 그룹 이름', self.target)
        self.shield = QCheckBox('Shield 사용: 충돌 예상 시 기준 경로 추종으로 대체')
        self.shield.setChecked(True)
        form.addRow(self.shield)
        self.device = QComboBox()
        self.device.addItems(['cpu', 'cuda', 'auto'])
        form.addRow('추론 장치', self.device)
        layout.addWidget(self.fields)
        self.status = QLabel('모델과 입력을 선택한 뒤 추론을 실행하세요.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(600)
        layout.addWidget(self.log)
        row = QHBoxLayout()
        self.start = QPushButton('추론 실행')
        self.cancel = QPushButton('중지')
        self.cancel.setEnabled(False)
        self.open_result = QPushButton('결과 불러오기')
        self.open_result.setEnabled(False)
        for button in (self.start, self.cancel, self.open_result):
            row.addWidget(button)
        layout.addLayout(row)
        self.start.clicked.connect(self._start)
        self.cancel.clicked.connect(self._cancel)
        self.open_result.clicked.connect(self._load_result)
        self.edits['model'].editingFinished.connect(self._model_config)
        self.edits['snapshot'].editingFinished.connect(self._snapshot_reference)
        models = list((ROOT / 'debug/rl').glob('*/final_model.zip'))
        if models:
            self.edits['model'].setText(str(max(models, key=lambda p: p.stat().st_mtime)))
            self._model_config()
        self.edits['snapshot'].setText(str(ROOT / 'sample/planning_snapshot_p3_6.apfgoal.pkl'))
        self._snapshot_reference()

    def _model_config(self):
        parent = Path(self.edits['model'].text()).parent
        if parent.name == 'checkpoints':
            parent = parent.parent
        if (parent / 'config.json').is_file():
            self.edits['config'].setText(str(parent / 'config.json'))

    def _snapshot_reference(self):
        match = re.search(r'_(p\d+)_', Path(self.edits['snapshot'].text()).name)
        if match:
            path = ROOT / 'runs_apfgoal' / f'RRTConnect_none_{match.group(1)}'
            if path.is_dir():
                self.edits['reference'].setText(str(path))

    def _browse(self, key):
        current = self.edits[key].text() or str(ROOT)
        if key == 'reference':
            path = QFileDialog.getExistingDirectory(self, '기준 경로 폴더 선택', current)
        else:
            extension = {'model': 'zip', 'config': 'json', 'snapshot': 'pkl'}[key]
            path, _ = QFileDialog.getOpenFileName(self, '파일 선택', current, f'Files (*.{extension})')
        if path:
            self.edits[key].setText(path)
            if key == 'model':
                self._model_config()
            elif key == 'snapshot':
                self._snapshot_reference()

    def _start(self):
        values = {k: Path(e.text().strip()).expanduser().resolve() for k, e in self.edits.items()}
        for key, path in values.items():
            required = path / 'summary.csv' if key == 'reference' else path
            if not required.is_file():
                QMessageBox.warning(self, '입력 확인', f'{key}: 파일을 찾을 수 없습니다.\n{required}')
                return
        output = ROOT / 'debug/rl_inference' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        args = ['-u', str(ROOT / 'python/rl/infer_sac.py')]
        for key, value in values.items():
            args += ['--' + key, str(value)]
        args += ['--out', str(output), '--device', self.device.currentText()]
        if self.shield.isChecked():
            args.append('--shield')
        if self.target.text().strip():
            args += ['--target', self.target.text().strip()]
        self.result = None
        self.cancelled = False
        self.pending = ''
        self.decoder.reset()
        self.log.clear()
        self._running(True)
        self.progress.setRange(0, 0)
        self.status.setText('추론 프로세스 시작 중…')
        env = QProcessEnvironment.systemEnvironment()
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            env.insert(name, '1')
        env.insert('PYTHONNOUSERSITE', '1')
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.start(sys.executable, args)

    def _running(self, running):
        self.fields.setEnabled(not running)
        self.start.setEnabled(not running)
        self.cancel.setEnabled(running)
        self.open_result.setEnabled(not running and bool(self.result))

    def _read_output(self):
        self.pending += self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            self.log.appendPlainText(line)
            if not line.startswith('SAC_UI '):
                continue
            try:
                event = json.loads(line[7:])
            except ValueError:
                continue
            kind = event.get('type')
            if kind == 'status':
                self.status.setText(event['message'])
            elif kind in ('progress', 'target'):
                self.progress.setRange(0, event['total'])
                self.progress.setValue(event['current'])
                self.status.setText(f"{event['current']}/{event['total']}: {event['target']}")
            elif kind == 'complete':
                self.result = event

    def _finished(self, code, exit_status):
        self._read_output()
        if code != 0 or exit_status != QProcess.ExitStatus.NormalExit or self.cancelled:
            self.result = None
        self._running(False)
        if self.result:
            r = self.result
            self.status.setText(f"완료: 성공 {r['n_success']}, 실패 {r['n_failed']}, 제외 {r['n_skipped']}\n"
                                f"{r['run_dir']}\n결과 불러오기 후 Start Simulation으로 재생하세요.")
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status.setText('추론을 중지했습니다.' if self.cancelled else '추론 실패: 아래 로그를 확인하세요.')

    def _error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.result = None
            self._running(False)
            self.progress.setRange(0, 1)
            self.status.setText('프로세스를 시작할 수 없습니다: ' + self.process.errorString())

    def _cancel(self):
        self.cancelled = True
        self.process.kill()

    def _load_result(self):
        if self.result:
            self.load_result.emit(self.result['run_dir'])
            self.accept()

    def closeEvent(self, event):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self._cancel()
            self.process.waitForFinished(1000)
        super().closeEvent(event)

    def reject(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self._cancel()
            self.process.waitForFinished(1000)
        super().reject()
