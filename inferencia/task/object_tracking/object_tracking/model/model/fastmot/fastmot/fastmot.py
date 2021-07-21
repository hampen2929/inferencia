import cv2
from pdb import Pdb
import numpy as np
from collections import OrderedDict

from .tracker import MultiTracker
from .utils import Profiler
from .utils.visualization import draw_tracks, draw_detections
from .utils.visualization import draw_flow_bboxes, draw_background_flow

from inferencia.task.object_detection.object_detection_2d.object_detection_2d_manager import ObjectDetection2DManager
from inferencia.task.person_reid.body_reid.body_reid_manager import BodyReidManager
from inferencia.util.logger.logger import Logger
from inferencia.task.object_tracking.object_tracking.model.object_tracking_result import ObjectTrackingResult
from inferencia.task.object_tracking.object_tracking.model.object_tracking_history import ObjectTrackingHistory


DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('label_name', 'U30'),
     ('conf', float)],
    align=True
)


# class ObjectTrackingHistory():

#     def __init__(self,
#                  max_hold_ret_num=100):
#         self.trk_ret = OrderedDict()
#         self.max_hold_ret_num = max_hold_ret_num
#         self.init()

#     def init(self):
#         self.trk_ret.clear()

#     def get_trk_ret_as_array(self, tracking_id):
#         if tracking_id not in self.trk_ret.keys():
#             msg = "trackin_id({}) does not exist.".format(tracking_id)
#             raise ValueError(msg)
#         return np.array(self.trk_ret[tracking_id])

#     def append(self,
#                tracking_id,
#                obj_det_ret):
#         if tracking_id not in self.trk_ret.keys():
#             self.trk_ret[tracking_id] = []
#         self.trk_ret[tracking_id].append(obj_det_ret)

#         if len(self.trk_ret[tracking_id]) > self.max_hold_ret_num:
#             del self.trk_ret[tracking_id][0]

#     def get_last_bboxes(self):
#         last_bboxes = []
#         for _, trk_ret in self.trk_ret.items():
#             last_bboxes.append(trk_ret[-1].to_list())
#         last_bboxes = np.array(last_bboxes)
#         return last_bboxes

#     def get_latest_trk_rets(self):
#         latest_trk_rets = {}
#         for trk_id in self.trk_ret.keys():
#             latest_trk_rets[trk_id] = self.trk_ret[trk_id][-1]
#         return latest_trk_rets


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
                 object_detection_model_name: str,
                 multi_tracker_config: dict,
                 use_iou_matching: bool,
                 use_feature_extractor: bool,
                 use_kalman_filter: bool,
                 feature_extractor_name: str,
                 input_fps: int,
                 target_fps: int,
                 max_hold_ret_num: int,
                 ):
        self.logger = Logger(__class__.__name__)
        init_msg = "\n===================== \n Initialize FastMOT \n=====================\n"
        self.logger.info(init_msg)

        self.size = (1280, 720)
        self.draw = False
        self.verbose = False

        self.detector = ObjectDetection2DManager.get_model(
            model_name=object_detection_model_name
        )

        self.extractor = BodyReidManager.get_model(
            model_name=feature_extractor_name
        )
        self.tracker = MultiTracker(size=self.size,
                                    metric="cosine",
                                    config=multi_tracker_config)
        self.frame_count = 0
        self.detector_frame_skip = round(input_fps / target_fps)
        self.obt_trk_hist = ObjectTrackingHistory(
            max_hold_ret_num=max_hold_ret_num)

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

    def step(self, frame, frame_index):
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
                     det_ret.class_name,
                     det_ret.confidence)
                )
            detections = np.asarray(det_post,
                                    dtype=DET_DTYPE).view(np.recarray)

            self.tracker.init(frame, detections)
        else:
            if self.frame_count % self.detector_frame_skip == 0:
                # with Profiler('compute_flow'):
                #     self.tracker.compute_flow(frame)

                with Profiler('detect'):
                    det_rets = self.detector.inference(frame)

                    det_post = []
                    for det_ret in det_rets:
                        det_post.append(
                            (np.array([det_ret.xmin, det_ret.ymin,
                                       det_ret.xmax, det_ret.ymax]),
                             det_ret.class_id,
                             det_ret.class_name,
                             det_ret.confidence)
                        )
                    detections = np.asarray(det_post,
                                            dtype=DET_DTYPE).view(np.recarray)

                with Profiler('apply_kalman', aggregate=True):
                    self.tracker.apply_kalman()

                with Profiler('extract'):
                    cropped_images = []
                    for det in detections:
                        bbox, _, _, _ = det
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
                    self.tracker.update(self.frame_count,
                                        detections,
                                        embeddings)
            else:
                with Profiler('track'):
                    self.tracker.track(frame)

        if self.draw:
            self._draw(frame, detections)
        self.frame_count += 1

        for tracking_id, trk_ret in self.tracker.tracks.items():
            xmin, ymin, xmax, ymax = trk_ret.tlbr
            obj_det_ret = ObjectTrackingResult(
                frame_index=frame_index,
                tracking_id=tracking_id,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                class_id=trk_ret.label,
                class_name=trk_ret.label_name,
                confidence=trk_ret.confidence,
                is_active=True
            )
            self.obt_trk_hist.append(tracking_id, obj_det_ret)

        obt_trk_hist_keys = self.obt_trk_hist.trk_ret.keys()
        only_hist_keys = list(set(obt_trk_hist_keys) -
                              set(self.tracker.tracks.keys()))

        for tracking_id in only_hist_keys:
            null_obj_det_ret = ObjectTrackingResult(
                frame_index=frame_index,
                tracking_id=tracking_id,
                xmin=np.nan,
                ymin=np.nan,
                xmax=np.nan,
                ymax=np.nan,
                class_id='',
                class_name=trk_ret.label_name,
                confidence=trk_ret.confidence,
                is_active=False
            )
            self.obt_trk_hist.append(tracking_id, null_obj_det_ret)

        # for tracking_id, track in self.tracker.tracks.items():
        #     self.tracking_results.append(tracking_id,

        #                                  obj_det_ret)

        return self.obt_trk_hist

    def print_timing_info(self):
        self.logger.info('=================Timing Stats=================')
        self.logger.info(
            f"{'track time:':<37}{Profiler.get_avg_millis('track'):>6.3f} ms")
        self.logger.info(
            f"{'preprocess time:':<37}{Profiler.get_avg_millis('preproc'):>6.3f} ms")
        self.logger.info(f"{'compute_flow time:':<37}"
                         f"{Profiler.get_avg_millis('compute_flow'):>6.3f} ms")
        self.logger.info(f"{'apply_kalman time:':<37}"
                         f"{Profiler.get_avg_millis('apply_kalman'):>6.3f} ms")
        self.logger.info(
            f"{'detect/flow time:':<37}{Profiler.get_avg_millis('detect'):>6.3f} ms")
        self.logger.info(f"{'feature extract time:':<37}"
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
