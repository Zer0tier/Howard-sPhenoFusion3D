import os
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

    @pyqtSlot(str, str, str, int)
    def on_run_clicked(self, rgb_dir, depth_dir, intrinsics_path, step_size):
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

        # Load intrinsics
        intr = load_intrinsics(intrinsics_path) if intrinsics_path else None
        if intr:
            K, dist, _, _ = intr
        else:
            K, dist = get_default_intrinsics()

        self.status_changed.emit(f'Starting reconstruction: {len(pairs)} frames...')

        is_icl = 'icl' in rgb_dir.lower()

        # --- Depth / camera parameters ---
        depth_scale = 5000.0 if is_icl else 1000.0
        depth_trunc = 4.0    if is_icl else 4.0   # 4 m covers the full gantry scene
        voxel_size  = 0.02   if is_icl else 0.005  # used for ICP radius and output downsampling

        # --- ICP-mode parameters (only used when use_known_poses=False) ---
        max_iter     = 30    if is_icl else 80
        bbox         = None  if is_icl else None   # no crop -- let TSDF use full frame
        erode        = True  if is_icl else False
        inpaint      = True  if is_icl else False
        depth_min_mm = 0     if is_icl else 0      # no near-clip; sparse data needs every pixel

        # --- Known-pose / TSDF parameters ---
        # use_known_poses=True skips ICP entirely and uses kinematic gantry poses.
        # gantry_step_m is per-PAIR displacement (= per-frame step × sampling step).
        # gantry_axis: 0=camera X (horizontal), 1=camera Y.
        # Run calibrate_gantry.py to refine these values for your specific dataset.
        use_known_poses = False if is_icl else True
        gantry_axis     = 0                         # horizontal scan (most common)
        per_frame_step  = 0.0 if is_icl else 0.00127  # ~1.27 mm/frame at 38mm/s, 30fps
        # step_size comes from the UI; multiply here so Reconstructor gets per-pair displacement
        gantry_step_m   = per_frame_step * step_size
        tsdf_voxel_m    = 0.003   # 3 mm voxels (only active when use_known_poses=True)

        self.worker = ProcessingWorker(
            pairs=pairs, K=K, dist=dist,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            voxel_size=voxel_size,
            max_iter=max_iter,
            bbox=bbox,
            gantry_step_m=gantry_step_m,
            gantry_axis=gantry_axis,
            depth_min_mm=depth_min_mm,
            erode=erode,
            inpaint=inpaint,
            use_known_poses=use_known_poses,
            tsdf_voxel_m=tsdf_voxel_m,
            save_path=os.path.join(os.path.dirname(rgb_dir), 'output')
        )
        self.worker.frame_done.connect(self._on_frame)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self.error_occurred)
        self.worker.start()
        self.viewer.start()

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
        self.viewer.update(pcd)
        self.frame_processed.emit(idx, total, pcd, fitness, rmse, status)
        self.status_changed.emit(f'Frame {idx + 1}/{total} | fitness={fitness:.4f}')

    @pyqtSlot(object, list, list)
    def _on_finished(self, final_pcd, succeed, fail):
        self.final_pcd = final_pcd
        self.status_changed.emit(
            f'Done. {len(succeed)} frames succeeded, {len(fail)} failed. '
            f'Use File menu to export.'
        )
        self.reconstruction_complete.emit(final_pcd, succeed, fail)

    def export_ply(self, path):
        if self.final_pcd:
            ok = save_ply(self.final_pcd, path)
            self.status_changed.emit(f'PLY saved: {path}' if ok else 'PLY export failed.')

    def export_csv(self, path):
        ok = save_metrics_csv(self.all_metrics, path)
        self.status_changed.emit(f'CSV saved: {path}' if ok else 'CSV export failed.')