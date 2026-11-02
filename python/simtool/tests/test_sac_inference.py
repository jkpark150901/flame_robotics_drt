import json
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QApplication
from rl.infer_sac import inference_config
from simtool.sac_inference_dialog import SACInferenceDialog


def test_saved_training_config_is_made_deterministic_without_changing_source(tmp_path):
    original = {'env': {'robot': 'dda', 'target_split': {'split': 'train'}, 'target_filter': ['old'],
                       'random_obstacles': {'enabled': True}, 'curriculum': {'enabled': True},
                       'residual': {'enabled': True, 'random_reference': True,
                                    'mid_start_prob': .3, 'start_noise_prob': .3}}, 'reward': {'apf': 10}}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'cfg': original}))
    cfg = inference_config(path, tmp_path / 'references', True, ['selected'])
    assert cfg['reward'] == original['reward']
    e = cfg['env']
    assert 'target_split' not in e
    assert e['target_filter'] == ['selected']
    assert not e['random_obstacles']['enabled'] and not e['curriculum']['enabled']
    assert e['residual']['shield'] and not e['residual']['random_reference']
    assert e['residual']['mid_start_prob'] == e['residual']['start_noise_prob'] == 0
    assert json.loads(path.read_text())['cfg'] == original


def test_rejects_non_residual_model_config(tmp_path):
    p = tmp_path / 'config.json'
    p.write_text(json.dumps({'env': {'residual': {'enabled': False}}}))
    with pytest.raises(ValueError, match='residual'):
        inference_config(p, tmp_path, False)


@pytest.fixture
def dialog():
    app = QApplication.instance() or QApplication([])
    d = SACInferenceDialog()
    yield d
    d.close()
    app.processEvents()


def complete_event(dialog):
    dialog.pending = 'SAC_UI ' + json.dumps({'type': 'complete', 'run_dir': '/tmp/test_inference',
                                          'n_success': 1, 'n_failed': 1, 'n_skipped': 0}) + '\n'


def test_only_completed_result_can_be_loaded(dialog):
    loaded = []
    dialog.load_result.connect(loaded.append)
    assert not dialog.open_result.isEnabled()
    complete_event(dialog)
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    assert dialog.open_result.isEnabled()
    assert '실패 1' in dialog.status.text()
    dialog._load_result()
    assert loaded == ['/tmp/test_inference']


@pytest.mark.parametrize('cancelled,code', [(True, 0), (False, 1)])
def test_failed_or_cancelled_process_never_loads_results(dialog, cancelled, code):
    complete_event(dialog)
    dialog.cancelled = cancelled
    dialog._finished(code, QProcess.ExitStatus.NormalExit)
    assert dialog.result is None and not dialog.open_result.isEnabled()


def test_failed_start_reenables_controls(dialog):
    dialog._running(True)
    dialog._error(QProcess.ProcessError.FailedToStart)
    assert dialog.start.isEnabled() and not dialog.cancel.isEnabled()
