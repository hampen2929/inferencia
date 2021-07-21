import numpy as np
from collections import OrderedDict


class ObjectTrackingHistory():

    def __init__(self, max_hold_ret_num):
        self.trk_ret = OrderedDict()
        self.max_hold_ret_num = max_hold_ret_num
        self.init()

    def init(self):
        self.trk_ret.clear()

    def get_trk_ret_as_array(self, tracking_id):
        if tracking_id not in self.trk_ret.keys():
            msg = "trackin_id({}) does not exist.".format(tracking_id)
            raise ValueError(msg)
        return np.array(self.trk_ret[tracking_id])

    def append(self,
               tracking_id,
               obj_det_ret):
        if tracking_id not in self.trk_ret.keys():
            self.trk_ret[tracking_id] = []
        self.trk_ret[tracking_id].append(obj_det_ret)

        if len(self.trk_ret[tracking_id]) > self.max_hold_ret_num:
            del self.trk_ret[tracking_id][0]

    def get_last_bboxes(self):
        last_bboxes = []
        for _, trk_ret in self.trk_ret.items():
            last_bboxes.append(trk_ret[-1].to_list())
        last_bboxes = np.array(last_bboxes)
        return last_bboxes

    def get_latest_track(self):
        latest_trk_rets = {}
        for trk_id in self.trk_ret.keys():
            if self.trk_ret[trk_id][-1].is_active:
                latest_trk_rets[trk_id] = self.trk_ret[trk_id][-1]
        return latest_trk_rets

    def get_track_history(self):
        return self.trk_ret
