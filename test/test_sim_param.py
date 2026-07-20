import sys
import json
import unittest

from PyQt6.QtWidgets import QApplication, QMainWindow, QSlider, QLineEdit

# Append path to import simtool successfully if needed
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
from simtool.param import SimParameterMap

class MockConsole:
    def info(self, msg): print("INFO:", msg)
    def error(self, msg): print("ERROR:", msg)

class MockUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.slider_positioner_x_pos = QSlider()
        self.slider_positioner_z_pos = QSlider()
        self.slider_positioner_r_pos = QSlider()

        self.edit_positioner_x_pos = QLineEdit()
        self.edit_positioner_z_pos = QLineEdit()
        self.edit_positioner_r_pos = QLineEdit()

class TestSimParameterMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)

    def setUp(self):
        self.ui = MockUI()
        self.config = {}
        self.console = MockConsole()
        self.param_map = SimParameterMap(self.ui, self.config, self.console)

    def test_load_parameters(self):
        # Create a temp json file
        test_json = "test_params.json"
        with open(test_json, "w") as f:
            f.write('''{
                "positioner_x_resolution": 0.01,
                "positioner_z_resolution": 0.02,
                "positioner_r_resolution": 0.5,
                "positioner_x_range": [0.0, 10.0],
                "positioner_z_range": [0.0, 5.0],
                "positioner_r_range": [-180.0, 180.0]
            }''')

        self.param_map.load_parameters(test_json)

        # Check ranges
        self.assertEqual(self.ui.slider_positioner_x_pos.maximum(), int(10.0 / 0.01))
        self.assertEqual(self.ui.slider_positioner_z_pos.maximum(), int(5.0 / 0.02))
        self.assertEqual(self.ui.slider_positioner_r_pos.maximum(), int(360.0 / 0.5))

        # Check signal connections: Slider -> LineEdit
        self.ui.slider_positioner_x_pos.setValue(500) # 500 * 0.01 = 5.0
        self.assertEqual(self.ui.edit_positioner_x_pos.text(), "5.000")

        self.ui.slider_positioner_z_pos.setValue(100) # 100 * 0.02 = 2.0
        self.assertEqual(self.ui.edit_positioner_z_pos.text(), "2.000")

        self.ui.slider_positioner_r_pos.setValue(540) # -180 + 540 * 0.5 = 90.0
        self.assertEqual(self.ui.edit_positioner_r_pos.text(), "90.000")

        # Check signal connections: LineEdit -> Slider
        self.ui.edit_positioner_x_pos.setText("3.14")
        self.ui.edit_positioner_x_pos.editingFinished.emit() # fake editing finished
        self.assertEqual(self.ui.slider_positioner_x_pos.value(), int(3.14 / 0.01))

        # Clean up
        os.remove(test_json)

if __name__ == '__main__':
    unittest.main()
