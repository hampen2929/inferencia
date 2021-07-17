import cv2
from pdb import Pdb
import numpy as np

from .tracker import MultiTracker
from .utils import Profiler
from .utils.visualization import draw_tracks, draw_detections
from .utils.visualization import draw_flow_bboxes, draw_background_flow

from inferencia.task.object_detection.object_detection_2d.object_detection_2d_manager import ObjectDetection2DManager
from inferencia.task.person_reid.body_reid.body_reid_manager import BodyReidManager
from inferencia.util.logger.logger import Logger

DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)


class FastMOT:
    """
    This is the top level module that integrates detection, feature extraction,
    and tracking together.
    Parameters
    ----------
    size : (int, int)
        Width and height of each frame.
    cap_dt : float
        Time interval in seconds between each captured frame.
    config : Dict
        Tracker configuration.
    draw : bool
        Flag to toggle visualization drawing.
    verbose : bool
        Flag to toggle output verbosity.
    """

    def __init__(self,
                 config,
                 input_fps,
                 target_fps,
                 ):
        self.logger = Logger(__class__.__name__)
        init_msg = "\n===================== \n Initialize FastMOT \n=====================\n"
        self.logger.info(init_msg)

        self.size = (1280, 720)
        self.draw = False
        self.verbose = False

        self.detector = ObjectDetection2DManager.get_model(
            model_name="TinyYoloV4")

        self.extractor = BodyReidManager.get_model()
        self.tracker = MultiTracker(size=self.size,
                                    metric="cosine",
                                    config=config['multi_tracker'])
        self.frame_count = 0
        self.detector_frame_skip = round(input_fps / target_fps)

    @ property
    def visible_tracks(self):
        # retrieve confirmed and active tracks from the tracker
        return [track for track in self.tracker.tracks.values()
                if track.confirmed and track.active]

    def reset(self, cap_dt):
        """
        Resets multiple object tracker. Must be called before `step`.
        Parameters
        ----------
        cap_dt : float
            Time interval in seconds between each frame.
        """
        self.frame_count = 0
        self.tracker.reset_dt(cap_dt)

    def step(self, frame):
        """
        Runs multiple object tracker on the next frame.
        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        detections = []
        if self.frame_count == 0:
            det_rets = self.detector.inference(frame)
            det_post = []
            for det_ret in det_rets:
                det_post.append(
                    (np.array([det_ret.xmin, det_ret.ymin,
                               det_ret.xmax, det_ret.ymax]),
                     det_ret.class_id,
                     det_ret.confidence)
                )
            detections = np.asarray(det_post,
                                    dtype=DET_DTYPE).view(np.recarray)

            self.tracker.init(frame, detections)
        else:
            if self.frame_count % self.detector_frame_skip == 0:
                with Profiler('detect'):
                    with Profiler('track'):
                        self.tracker.compute_flow(frame)
                    det_rets = self.detector.inference(frame)

                    det_post = []
                    for det_ret in det_rets:
                        det_post.append(
                            (np.array([det_ret.xmin, det_ret.ymin,
                                       det_ret.xmax, det_ret.ymax]),
                             det_ret.class_id,
                             det_ret.confidence)
                        )
                    detections = np.asarray(det_post,
                                            dtype=DET_DTYPE).view(np.recarray)

                with Profiler('extract'):
                    with Profiler('track', aggregate=True):
                        self.tracker.apply_kalman()
                    cropped_images = []
                    for det in detections:
                        bbox, _, _ = det
                        xmin, ymin, xmax, ymax = bbox.astype(int)
                        cropped_image = frame[ymin: ymax,
                                              xmin: xmax]
                        cropped_images.append(cropped_image)
                    reid_rets = self.extractor.inference(cropped_images)
                    embeddings = []
                    for reid_ret in reid_rets:
                        embeddings.append(reid_ret.feature)
                    embeddings = np.array(embeddings)

                with Profiler('assoc'):
                    self.tracker.update(
                        self.frame_count, detections, embeddings)
            else:
                with Profiler('track'):
                    self.tracker.track(frame)

            # import pdb
            # pdb.set_trace()
            print(self.tracker.tracks)

            # detections
            # <class 'numpy.recarray'>
            # rec.array([([208., 300., 570., 609.], 1, 0.98401898)],dtype={'names':['tlbr','label','conf'], 'formats':[('<f8', (4,)),'<i8','<f8'], 'offsets':[0,32,40], 'itemsize':48, 'aligned':True})

            # embeddings
            # (1, 512)
            # float32
            # np.ndarray

        if self.draw:
            self._draw(frame, detections)
        self.frame_count += 1

    def print_timing_info(self):
        self.logger.info('=================Timing Stats=================')
        self.logger.info(
            f"{'track time:':<37}{Profiler.get_avg_millis('track'):>6.3f} ms")
        self.logger.info(
            f"{'preprocess time:':<37}{Profiler.get_avg_millis('preproc'):>6.3f} ms")
        self.logger.info(
            f"{'detect/flow time:':<37}{Profiler.get_avg_millis('detect'):>6.3f} ms")
        self.logger.info(f"{'feature extract/kalman filter time:':<37}"
                         f"{Profiler.get_avg_millis('extract'):>6.3f} ms")
        self.logger.info(
            f"{'association time:':<37}{Profiler.get_avg_millis('assoc'):>6.3f} ms")

    def _draw(self, frame, detections):
        draw_tracks(frame, self.visible_tracks, show_flow=self.verbose)
        if self.verbose:
            draw_detections(frame, detections)
            draw_flow_bboxes(frame, self.tracker)
            # draw_background_flow(frame, self.tracker)
        cv2.putText(frame, f'visible: {len(self.visible_tracks)}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
