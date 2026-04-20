"""
app/capture_worker.py
---------------------
QThread that drives a CaptureBackend off the UI thread.
"""

from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal

from capture import CaptureParams, get_backend, go_home


class CaptureWorker(QThread):

    frame_captured = pyqtSignal(int, int)        # idx, total_estimate (0=unknown)
    finished       = pyqtSignal(str, int)        # out_dir, n_frames
    error          = pyqtSignal(str)

    def __init__(self, backend_pref: str, params: CaptureParams):
        super().__init__()
        self.backend_pref = backend_pref
        self.params       = params
        self._backend     = None

    def run(self):
        try:
            self._backend = get_backend(self.backend_pref)
            out_dir = self._backend.start(
                self.params,
                on_progress=lambda i, t: self.frame_captured.emit(i, t),
                on_done=lambda d, n: self.finished.emit(d, n),
                on_error=lambda msg: self.error.emit(msg),
            )
            # Note: on_done is called inside backend.start(); nothing else to do
            if out_dir is None:
                # error already emitted via on_error
                return
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        if self._backend is not None:
            self._backend.stop()


class HomeWorker(QThread):
    """Drive the gantry back to its start position off the UI thread."""

    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, target_position_m: float = 0.005,
                 velocity_mps: float = 0.2):
        super().__init__()
        self.target_position_m = target_position_m
        self.velocity_mps      = velocity_mps

    def run(self):
        try:
            go_home(
                target_position_m=self.target_position_m,
                velocity_mps=self.velocity_mps,
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
