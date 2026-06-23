import json

class SimParameterMap:
    def __init__(self, ui, config, console):
        self.ui = ui
        self.config = config
        self.console = console

        self.x_res = 0.01
        self.z_res = 0.01
        self.r_res = 0.1
        self.clamp_res = 0.01
        self.spool_x_res = 0.01
        self.spool_y_res = 0.01
        self.spool_z_res = 0.01
        self.spool_x_rot_res = 0.1
        self.spool_z_rot_res = 0.1
        self.spool_x_range = [0.0, 10.0]
        self.spool_y_range = [0.0, 5.0]
        self.spool_z_range = [0.0, 5.0]
        self.spool_x_rot_range = [0.0, 360.0]
        self.spool_z_rot_range = [0.0, 360.0]

        self._bind_signals()
        self.load_parameters(self.config)

    def _bind_signals(self):
        """Bind UI signals for real-time synchronization between sliders and line edits"""
        
        # Sliders
        if hasattr(self.ui, 'slider_positioner_x_pos'):
            self.ui.slider_positioner_x_pos.valueChanged.connect(self._on_slider_x_changed)
        if hasattr(self.ui, 'slider_positioner_z_pos'):
            self.ui.slider_positioner_z_pos.valueChanged.connect(self._on_slider_z_changed)
        if hasattr(self.ui, 'slider_positioner_r_pos'):
            self.ui.slider_positioner_r_pos.valueChanged.connect(self._on_slider_r_changed)

        if hasattr(self.ui, 'slider_positioner_clamp_pos'):
            self.ui.slider_positioner_clamp_pos.valueChanged.connect(self._on_slider_clamp_changed)
        if hasattr(self.ui, 'slider_spool_x_pos'):
            self.ui.slider_spool_x_pos.valueChanged.connect(self._on_slider_spool_x_changed)
        if hasattr(self.ui, 'slider_spool_y_pos'):
            self.ui.slider_spool_y_pos.valueChanged.connect(self._on_slider_spool_y_changed)
        if hasattr(self.ui, 'slider_spool_z_pos'):
            self.ui.slider_spool_z_pos.valueChanged.connect(self._on_slider_spool_z_changed)
        if hasattr(self.ui, 'slider_spool_x_rot'):
            self.ui.slider_spool_x_rot.valueChanged.connect(self._on_slider_spool_x_rot_changed)
        if hasattr(self.ui, 'slider_spool_z_rot'):
            self.ui.slider_spool_z_rot.valueChanged.connect(self._on_slider_spool_z_rot_changed)

        # LineEdits
        if hasattr(self.ui, 'edit_positioner_x_pos'):
            self.ui.edit_positioner_x_pos.editingFinished.connect(self._on_edit_x_changed)
        if hasattr(self.ui, 'edit_positioner_z_pos'):
            self.ui.edit_positioner_z_pos.editingFinished.connect(self._on_edit_z_changed)
        if hasattr(self.ui, 'edit_positioner_r_pos'):
            self.ui.edit_positioner_r_pos.editingFinished.connect(self._on_edit_r_changed)
        if hasattr(self.ui, 'edit_positioner_clamp_pos'):
            self.ui.edit_positioner_clamp_pos.editingFinished.connect(self._on_edit_clamp_changed)
        if hasattr(self.ui, 'edit_spool_x_pos'):
            self.ui.edit_spool_x_pos.editingFinished.connect(self._on_edit_spool_x_changed)
        if hasattr(self.ui, 'edit_spool_y_pos'):
            self.ui.edit_spool_y_pos.editingFinished.connect(self._on_edit_spool_y_changed)
        if hasattr(self.ui, 'edit_spool_z_pos'):
            self.ui.edit_spool_z_pos.editingFinished.connect(self._on_edit_spool_z_changed)
        if hasattr(self.ui, 'edit_spool_x_rot'):
            self.ui.edit_spool_x_rot.editingFinished.connect(self._on_edit_spool_x_rot_changed)
        if hasattr(self.ui, 'edit_spool_z_rot'):
            self.ui.edit_spool_z_rot.editingFinished.connect(self._on_edit_spool_z_rot_changed)

    def load_parameters(self, source):
        """Load simulation parameters from a file path or a dict."""
        try:
            if isinstance(source, dict):
                self._update_ui_parameters(source)
            else:
                with open(source, 'r', encoding='utf-8') as f:
                    sim_params = json.load(f)
                    self.console.info(f"Loaded simulation parameters from {source}")
                    self._update_ui_parameters(sim_params)
        except Exception as e:
            self.console.error(f"Failed to load parameters: {e}")

    def _update_ui_parameters(self, params):
        """Update the range and resolution of the sliders based on loaded parameters"""
        self.x_res = params.get('positioner_x_resolution', 0.01)
        self.z_res = params.get('positioner_z_resolution', 0.01)
        self.r_res = params.get('positioner_r_resolution', 0.1)
        self.clamp_res = params.get('positioner_clamp_resolution', 0.01)
        self.spool_x_res = params.get('spool_x_resolution', 0.01)
        self.spool_y_res = params.get('spool_y_resolution', 0.01)
        self.spool_z_res = params.get('spool_z_resolution', 0.01)
        self.spool_x_rot_res = params.get('spool_x_rotation_resolution', 0.1)
        self.spool_z_rot_res = params.get('spool_z_rotation_resolution', 0.1)

        x_range = params.get('positioner_x_range', [0.0, 8.0])
        z_range = params.get('positioner_z_range', [0.0, 3.0])
        r_range = params.get('positioner_r_range', [0.0, 360.0])
        clamp_range = params.get('positioner_clamp_range', [0.0, 0.9])
        self.spool_x_range = params.get('spool_x_range', [0.0, 10.0])
        self.spool_y_range = params.get('spool_y_range', [0.0, 5.0])
        self.spool_z_range = params.get('spool_z_range', [0.0, 5.0])
        self.spool_x_rot_range = params.get('spool_x_rotation_range', [0.0, 360.0])
        self.spool_z_rot_range = params.get('spool_z_rotation_range', [0.0, 360.0])

        if hasattr(self.ui, 'slider_positioner_x_pos'):
            self.ui.slider_positioner_x_pos.setMinimum(0)
            self.ui.slider_positioner_x_pos.setMaximum(int((x_range[1] - x_range[0]) / self.x_res))

        if hasattr(self.ui, 'slider_positioner_z_pos'):
            self.ui.slider_positioner_z_pos.setMinimum(0)
            self.ui.slider_positioner_z_pos.setMaximum(int((z_range[1] - z_range[0]) / self.z_res))

        if hasattr(self.ui, 'slider_positioner_r_pos'):
            self.ui.slider_positioner_r_pos.setMinimum(0)
            self.ui.slider_positioner_r_pos.setMaximum(int((r_range[1] - r_range[0]) / self.r_res))

        if hasattr(self.ui, 'slider_positioner_clamp_pos'):
            self.ui.slider_positioner_clamp_pos.setMinimum(0)
            self.ui.slider_positioner_clamp_pos.setMaximum(int((clamp_range[1] - clamp_range[0]) / self.clamp_res))

        self._configure_spool_slider('x', self.spool_x_range, self.spool_x_res)
        self._configure_spool_slider('y', self.spool_y_range, self.spool_y_res)
        self._configure_spool_slider('z', self.spool_z_range, self.spool_z_res)
        self._configure_spool_rotation_slider('x')
        self._configure_spool_rotation_slider('z')

        default_pos = params.get('spool_position_default', [7.311, 1.877, 1.213])
        self._set_spool_value('x', default_pos[0], update_slider=True)
        self._set_spool_value('y', default_pos[1], update_slider=True)
        self._set_spool_value('z', default_pos[2], update_slider=True)
        self.set_spool_rotation_value('x', params.get('spool_x_rotation_default', 0.0), update_slider=True)
        self.set_spool_rotation_value('z', params.get('spool_z_rotation_default', 0.0), update_slider=True)

    def _configure_spool_slider(self, axis, value_range, resolution):
        slider = getattr(self.ui, f'slider_spool_{axis}_pos', None)
        if slider is not None:
            slider.setMinimum(0)
            slider.setMaximum(int((value_range[1] - value_range[0]) / resolution))

    def _configure_spool_rotation_slider(self, axis):
        slider = getattr(self.ui, f'slider_spool_{axis}_rot', None)
        if slider is not None:
            value_range = getattr(self, f'spool_{axis}_rot_range')
            resolution = getattr(self, f'spool_{axis}_rot_res')
            slider.setMinimum(0)
            slider.setMaximum(int((value_range[1] - value_range[0]) / resolution))

    def _set_spool_value(self, axis, value, update_slider=False):
        line_edit = getattr(self.ui, f'edit_spool_{axis}_pos', None)
        if line_edit is not None:
            line_edit.setText(f"{value:.3f}")
        if update_slider:
            slider = getattr(self.ui, f'slider_spool_{axis}_pos', None)
            value_range = getattr(self, f'spool_{axis}_range')
            resolution = getattr(self, f'spool_{axis}_res')
            if slider is not None:
                slider.setValue(int((value - value_range[0]) / resolution))

    def set_spool_pose_values(self, x, y, z, x_rotation=None, z_rotation=None, update_sliders=True):
        self._set_spool_value('x', float(x), update_slider=update_sliders)
        self._set_spool_value('y', float(y), update_slider=update_sliders)
        self._set_spool_value('z', float(z), update_slider=update_sliders)
        if x_rotation is not None:
            self.set_spool_rotation_value('x', float(x_rotation), update_slider=update_sliders)
        if z_rotation is not None:
            self.set_spool_rotation_value('z', float(z_rotation), update_slider=update_sliders)

    def set_spool_rotation_value(self, axis, value, update_slider=False):
        line_edit = getattr(self.ui, f'edit_spool_{axis}_rot', None)
        if line_edit is not None:
            line_edit.setText(f"{value:.3f}")
        if update_slider:
            slider = getattr(self.ui, f'slider_spool_{axis}_rot', None)
            value_range = getattr(self, f'spool_{axis}_rot_range')
            resolution = getattr(self, f'spool_{axis}_rot_res')
            if slider is not None:
                slider.setValue(int((value - value_range[0]) / resolution))

    def set_positioner_values(self, x=None, z=None, r=None, clamp=None, update_sliders=True):
        values = {
            "x": (x, self.x_res, "edit_positioner_x_pos", "slider_positioner_x_pos"),
            "z": (z, self.z_res, "edit_positioner_z_pos", "slider_positioner_z_pos"),
            "r": (r, self.r_res, "edit_positioner_r_pos", "slider_positioner_r_pos"),
            "clamp": (clamp, self.clamp_res, "edit_positioner_clamp_pos", "slider_positioner_clamp_pos"),
        }
        for _, (value, resolution, edit_name, slider_name) in values.items():
            if value is None:
                continue
            value = float(value)
            line_edit = getattr(self.ui, edit_name, None)
            if line_edit is not None:
                line_edit.setText(f"{value:.3f}")
            if update_sliders:
                slider = getattr(self.ui, slider_name, None)
                if slider is not None:
                    slider.setValue(int(value / resolution))

    # --- Handlers for Sliders -> LineEdits ---

    def _on_slider_x_changed(self, value):
        if hasattr(self.ui, 'edit_positioner_x_pos'):
            real_val = value * self.x_res
            self.ui.edit_positioner_x_pos.setText(f"{real_val:.3f}")

    def _on_slider_z_changed(self, value):
        if hasattr(self.ui, 'edit_positioner_z_pos'):
            real_val = value * self.z_res
            self.ui.edit_positioner_z_pos.setText(f"{real_val:.3f}")

    def _on_slider_r_changed(self, value):
        if hasattr(self.ui, 'edit_positioner_r_pos'):
            real_val = value * self.r_res
            self.ui.edit_positioner_r_pos.setText(f"{real_val:.3f}")

    def _on_slider_clamp_changed(self, value):
        if hasattr(self.ui, 'edit_positioner_clamp_pos'):
            real_val = value * self.clamp_res
            self.ui.edit_positioner_clamp_pos.setText(f"{real_val:.3f}")

    def _on_slider_spool_x_changed(self, value):
        self._set_spool_value('x', self.spool_x_range[0] + value * self.spool_x_res)

    def _on_slider_spool_y_changed(self, value):
        self._set_spool_value('y', self.spool_y_range[0] + value * self.spool_y_res)

    def _on_slider_spool_z_changed(self, value):
        self._set_spool_value('z', self.spool_z_range[0] + value * self.spool_z_res)

    def _on_slider_spool_x_rot_changed(self, value):
        self.set_spool_rotation_value('x', self.spool_x_rot_range[0] + value * self.spool_x_rot_res)

    def _on_slider_spool_z_rot_changed(self, value):
        self.set_spool_rotation_value('z', self.spool_z_rot_range[0] + value * self.spool_z_rot_res)

    # --- Handlers for LineEdits -> Sliders ---

    def _on_edit_x_changed(self):
        if hasattr(self.ui, 'edit_positioner_x_pos') and hasattr(self.ui, 'slider_positioner_x_pos'):
            try:
                val = float(self.ui.edit_positioner_x_pos.text())
                self.ui.slider_positioner_x_pos.setValue(int(val / self.x_res))
            except ValueError:
                pass

    def _on_edit_z_changed(self):
        if hasattr(self.ui, 'edit_positioner_z_pos') and hasattr(self.ui, 'slider_positioner_z_pos'):
            try:
                val = float(self.ui.edit_positioner_z_pos.text())
                self.ui.slider_positioner_z_pos.setValue(int(val / self.z_res))
            except ValueError:
                pass

    def _on_edit_r_changed(self):
        if hasattr(self.ui, 'edit_positioner_r_pos') and hasattr(self.ui, 'slider_positioner_r_pos'):
            try:
                val = float(self.ui.edit_positioner_r_pos.text())
                self.ui.slider_positioner_r_pos.setValue(int(val / self.r_res))
            except ValueError:
                pass

    def _on_edit_clamp_changed(self):
        if hasattr(self.ui, 'edit_positioner_clamp_pos') and hasattr(self.ui, 'slider_positioner_clamp_pos'):
            try:
                val = float(self.ui.edit_positioner_clamp_pos.text())
                self.ui.slider_positioner_clamp_pos.setValue(int(val / self.clamp_res))
            except ValueError:
                pass

    def _on_edit_spool_x_changed(self):
        self._sync_spool_edit_to_slider('x')

    def _on_edit_spool_y_changed(self):
        self._sync_spool_edit_to_slider('y')

    def _on_edit_spool_z_changed(self):
        self._sync_spool_edit_to_slider('z')

    def _on_edit_spool_x_rot_changed(self):
        self._sync_spool_rotation_edit_to_slider('x')

    def _on_edit_spool_z_rot_changed(self):
        self._sync_spool_rotation_edit_to_slider('z')

    def _sync_spool_edit_to_slider(self, axis):
        line_edit = getattr(self.ui, f'edit_spool_{axis}_pos', None)
        slider = getattr(self.ui, f'slider_spool_{axis}_pos', None)
        if line_edit is None or slider is None:
            return
        try:
            value = float(line_edit.text())
            value_range = getattr(self, f'spool_{axis}_range')
            resolution = getattr(self, f'spool_{axis}_res')
            slider.setValue(int((value - value_range[0]) / resolution))
        except ValueError:
            pass

    def _sync_spool_rotation_edit_to_slider(self, axis):
        line_edit = getattr(self.ui, f'edit_spool_{axis}_rot', None)
        slider = getattr(self.ui, f'slider_spool_{axis}_rot', None)
        if line_edit is None or slider is None:
            return
        try:
            value = float(line_edit.text())
            value_range = getattr(self, f'spool_{axis}_rot_range')
            resolution = getattr(self, f'spool_{axis}_rot_res')
            slider.setValue(int((value - value_range[0]) / resolution))
        except ValueError:
            pass
