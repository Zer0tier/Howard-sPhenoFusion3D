import os
import time
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from file_io.loader   import load_image_pairs, load_intrinsics, get_default_intrinsics
from file_io.exporter import save_ply, save_metrics_csv
from app.worker       import ProcessingWorker
from visualiser.viewer import PointCloudViewer


class Controller(QObject):

    status_changed         = pyqtSignal(str)
    frame_processed        = pyqtSignal(int, int, object, float, float, str)
    reconstruction_complete = pyqtSignal(object, list, list)
    error_occurred         = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker    = None
        self.viewer    = PointCloudViewer()
        self.final_pcd = None
        self.all_metrics = []
        self.n_success = 0
        self.n_fail    = 0
        self._viewer_update_stride = 3
        self._last_status_ts = 0.0

    @pyqtSlot(str, str, str, int, str)
    def on_run_clicked(self, rgb_dir, depth_dir, intrinsics_path, step_size, profile='balanced'):
        self.n_success   = 0
        self.n_fail      = 0
        self.all_metrics = []
        self.final_pcd   = None

        # Load pairs
        try:
            pairs = load_image_pairs(rgb_dir, depth_dir, step=step_size)
        except Exception as e:
            self.error_occurred.emit(f'Failed to load images:\n{e}')
            return
        if not pairs:
            self.error_occurred.emit('No RGB-D image pairs were found.')
            return

        # Load intrinsics
        intr = load_intrinsics(intrinsics_path) if intrinsics_path else None
        if intr:
            K, dist, _, _ = intr
            self.status_changed.emit('Loaded intrinsics from file.')
        else:
            K, dist = get_default_intrinsics()
            self.status_changed.emit('Intrinsics file missing/invalid. Using defaults.')

        self.status_changed.emit(
            f'Starting reconstruction: {len(pairs)} frames (step={step_size}, profile={profile}).'
        )

        # Detect depth scale from path hint
        depth_scale = self._infer_depth_scale(rgb_dir, depth_dir)
        perf = self._profile_settings(profile, step_size)
        self._viewer_update_stride = perf['viewer_stride']

        self.worker = ProcessingWorker(
            pairs=pairs, K=K, dist=dist,
            depth_scale=depth_scale,
            save_path=os.path.join(os.path.dirname(rgb_dir), 'output'),
            save_every_n_frames=perf['save_every_n_frames'],
            emit_pcd_every_n_frames=self._viewer_update_stride,
            icp_max_points=perf['icp_max_points'],
            fitness_threshold=perf['fitness_threshold'],
            voxel_size=perf['voxel_size'],
        )
        self.worker.frame_done.connect(self._on_frame)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self.error_occurred)
        self.worker.start()
        self.viewer.close()
        self.viewer.start()
        self._last_status_ts = 0.0

    @pyqtSlot()
    def on_stop_clicked(self):
        if self.worker:
            self.worker.stop()
        self.status_changed.emit('Stopping...')

    @pyqtSlot(int, int, object, float, float, str)
    def _on_frame(self, idx, total, pcd, fitness, rmse, status):
        if status == 'OK':
            self.n_success += 1
        else:
            self.n_fail += 1
        self.all_metrics.append({'frame': idx, 'status': status, 'fitness': fitness, 'rmse': rmse})
        if status == 'OK' and pcd is not None:
            self.viewer.update(pcd)
        self.frame_processed.emit(idx, total, pcd, fitness, rmse, status)
        now = time.time()
        if (now - self._last_status_ts) > 0.15 or idx == total - 1:
            self.status_changed.emit(
                f'Frame {idx + 1}/{total} | ok={self.n_success} fail={self.n_fail} | '
                f'fitness={fitness:.4f} rmse={rmse:.5f}'
            )
            self._last_status_ts = now

    @pyqtSlot(object, list, list)
    def _on_finished(self, final_pcd, succeed, fail):
        self.final_pcd = final_pcd
        self.status_changed.emit(
            f'Done. {len(succeed)} frames succeeded, {len(fail)} failed. '
            f'Use File menu to export.'
        )
        self.reconstruction_complete.emit(final_pcd, succeed, fail)

    def _infer_depth_scale(self, rgb_dir, depth_dir):
        joined = f'{rgb_dir} {depth_dir}'.lower()
        if any(tag in joined for tag in ('icl', 'nuim', 'tum', 'freiburg')):
            return 1.0
        return 1000.0

    def _profile_settings(self, profile, step_size):
        profile = (profile or 'balanced').lower()
        if profile == 'fast':
            return {
                'viewer_stride': 5,
                'icp_max_points': 25000 if step_size <= 2 else 18000,
                'fitness_threshold': 1e-5,
                'save_every_n_frames': 0,
                'voxel_size': 0.008,
            }
        if profile == 'quality':
            return {
                'viewer_stride': 1,
                'icp_max_points': 90000 if step_size <= 2 else 70000,
                'fitness_threshold': 1e-6,
                'save_every_n_frames': 0,
                'voxel_size': 0.004,
            }
        return {
            'viewer_stride': 3,
            'icp_max_points': 45000 if step_size <= 2 else 30000,
            'fitness_threshold': 1e-6,
            'save_every_n_frames': 0,
            'voxel_size': 0.005,
        }

    def export_ply(self, path):
        if self.final_pcd:
            ok = save_ply(self.final_pcd, path)
            self.status_changed.emit(f'PLY saved: {path}' if ok else 'PLY export failed.')

    def export_csv(self, path):
        ok = save_metrics_csv(self.all_metrics, path)
        self.status_changed.emit(f'CSV saved: {path}' if ok else 'CSV export failed.')