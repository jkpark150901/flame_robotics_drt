"""Time-only navigation; reference and recording clocks stay independent."""
import math
import tkinter as tk
from tkinter import ttk, messagebox


def time_groups(figure):
    groups = {}
    for ax in figure.axes:
        label = ax.get_xlabel().lower()
        if '[s]' not in label or not ('time' in label or '시간' in label):
            continue
        key = ax.get_xlabel().split('[s]')[0].strip()
        group = groups.setdefault(key, [])
        for candidate in figure.axes:
            if candidate is ax or ax.get_shared_x_axes().joined(ax, candidate):
                if candidate not in group:
                    group.append(candidate)
    return groups


def set_time_limits(axes, start, end):
    start, end = float(start), float(end)
    if not math.isfinite(start) or not math.isfinite(end) or start >= end:
        raise ValueError('시작·끝은 유한한 숫자이며 시작 < 끝이어야 합니다.')
    for ax in axes:
        ax.set_xlim(start, end)


class TimeAxisControls(ttk.Frame):
    def __init__(self, parent, canvas, toolbar=None):
        super().__init__(parent)
        self.canvas, self.toolbar = canvas, toolbar
        self.groups, self.full, self.callbacks = {}, {}, []
        self.views = {}
        self.key = None
        self._syncing = False
        self.clock = tk.StringVar()
        self.start, self.end = tk.StringVar(), tk.StringVar()
        ttk.Label(self, text='시간축').pack(side='left')
        self.selector = ttk.Combobox(self, textvariable=self.clock, state='readonly', width=23)
        self.selector.pack(side='left', padx=3)
        self.selector.bind('<<ComboboxSelected>>', lambda e: self._read_limits())
        self.widgets = []
        for label, var in [('시작', self.start), ('끝', self.end)]:
            ttk.Label(self, text=label).pack(side='left')
            entry = ttk.Entry(self, textvariable=var, width=8)
            entry.pack(side='left', padx=2)
            entry.bind('<Return>', lambda e: self.apply())
            self.widgets.append(entry)
        for label, fn in [('적용', self.apply), ('◀', lambda: self.shift(-1)),
                          ('▶', lambda: self.shift(1)), ('전체', self.reset)]:
            button = ttk.Button(self, text=label, command=fn, width=5)
            button.pack(side='left', padx=1)
            self.widgets.append(button)
        self.canvas.mpl_connect('scroll_event', self._scroll)
        self.refresh(None)

    def refresh(self, key):
        previous = dict(self.views) if key == self.key else {}
        old_full = self.full if key == self.key else {}
        for ax, cid in self.callbacks:
            ax.callbacks.disconnect(cid)
        self.callbacks = []
        self.key = key
        self.groups = time_groups(self.canvas.figure)
        self.full = {k: old_full.get(k, (min(a.get_xlim()[0] for a in axes),
                                       max(a.get_xlim()[1] for a in axes)))
                     for k, axes in self.groups.items()}
        self.views = {}
        for k, axes in self.groups.items():
            set_time_limits(axes, *previous.get(k, self.full[k]))
            self.views[k] = axes[0].get_xlim()
            for ax in axes:
                cid = ax.callbacks.connect('xlim_changed', lambda changed, name=k: self._changed(name, changed))
                self.callbacks.append((ax, cid))
        self.selector['values'] = list(self.groups)
        if self.clock.get() not in self.groups:
            self.clock.set(next(iter(self.groups), '시간 데이터 없음'))
        for widget in self.widgets:
            widget.configure(state='normal' if self.groups else 'disabled')
        if self.toolbar:
            self.toolbar.update()
        self._read_limits()

    def _changed(self, name, source):
        if self._syncing:
            return
        self._syncing = True
        self.views[name] = source.get_xlim()
        try:
            for ax in self.groups[name]:
                if ax is not source:
                    ax.set_xlim(source.get_xlim(), emit=False)
            if name == self.clock.get():
                self._read_limits()
        finally:
            self._syncing = False

    def _read_limits(self):
        axes = self.groups.get(self.clock.get())
        if axes:
            a, b = axes[0].get_xlim()
            self.start.set(f'{a:.6g}')
            self.end.set(f'{b:.6g}')
        else:
            self.start.set('')
            self.end.set('')

    def _set(self, start, end):
        axes = self.groups.get(self.clock.get())
        if not axes:
            return
        # Validate before recording navigation history.
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or start >= end:
            raise ValueError('시작·끝은 유한한 숫자이며 시작 < 끝이어야 합니다.')
        if self.toolbar:
            self.toolbar.push_current()
        set_time_limits(axes, start, end)
        if self.toolbar:
            self.toolbar.push_current()
        self.canvas.draw_idle()

    def apply(self):
        try:
            self._set(self.start.get(), self.end.get())
        except ValueError as exc:
            messagebox.showerror('시간 범위', str(exc), parent=self)

    def reset(self):
        if self.clock.get() in self.full:
            self._set(*self.full[self.clock.get()])

    def shift(self, direction):
        axes = self.groups.get(self.clock.get())
        if axes:
            a, b = axes[0].get_xlim()
            delta = (b-a) * .5 * direction
            self._set(a+delta, b+delta)

    def _scroll(self, event):
        if event.xdata is None or (self.toolbar and self.toolbar.mode):
            return
        for name, axes in self.groups.items():
            if event.inaxes in axes:
                self.clock.set(name)
                a, b = axes[0].get_xlim()
                factor = .8 if event.button == 'up' else 1.25
                x = event.xdata
                self._set(x+(a-x)*factor, x+(b-x)*factor)
                return
