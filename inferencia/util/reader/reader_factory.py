import os.path as osp
from .reader.video_reader import VideoReader


class ReaderFactory():
    video_exts = [".mp4", ".avi", ".mov", ".MOV", ".mkv"]

    def create(target_input):
        if osp.isfile(target_input):
            ext = osp.splitext(target_input)[1]
            if ext in ReaderFactory.video_exts:
                return VideoReader(target_input)
            else:
                msg = "{} is not supported. {} are supported.".format(
                    ext, ReaderFactory.video_exts)
                raise TypeError(msg)

        # elif osp.isdir(target_input):
        #     return ImageReader(target_input)

        # # USB camera
        # elif isinstance(target_input, int):
        #     return VideoReader(target_input)

        # # network camera
        # elif isinstance(target_input, str):
        #     return NetworkCameraReader(target_input)

        else:
            raise ValueError()
