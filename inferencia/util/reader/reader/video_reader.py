import cv2

from inferencia.util.formatter.format_frame_index import format_frame_index
from inferencia.util.logger.logger import Logger
from ..frame_data import FrameData
from ..base_reader import BaseReader


class VideoReader(BaseReader):

    def __init__(self, input_path):
        self.logger = Logger(__class__.__name__)
        init_msg = "\n===================== \n Initialize Reader \n=====================\n"
        self.logger.info(init_msg)

        self.cap = cv2.VideoCapture(input_path)
        self.__frame_index = -1
        self.__is_open = True
        self.__break_frame_index = 1000000000000

    def is_open(self):
        return self.__is_open

    def read(self):
        # for frame_num in range(len(self.file_paths[self.frame_index])):
        ret, frame = self.cap.read()
        if self.__break_frame_index <= self.__frame_index:
            ret = False
        if not ret:
            self.is_open = False
        self.__frame_index += 1
        frame_index_str = format_frame_index(self.__frame_index)
        # self.logger.info('frame_index is {}.'.format(self.__frame_index))
        frame_data = FrameData(ret=ret,
                               frame=frame,
                               frame_height=self.height,
                               frame_width=self.width,
                               frame_index=self.__frame_index,
                               frame_index_str=frame_index_str,
                               frame_path=None)
        return frame_data

    def forward_frame_index(self, frame_index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.__frame_index = frame_index

    def set_break_frame_index(self, break_frame_index):
        self.__break_frame_index = break_frame_index
        self.logger.info(
            'set_break_frame_index as {}.'.format(break_frame_index))

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
