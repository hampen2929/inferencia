from .reader_factory import ReaderFactory


class ReaderManager:
    def get_reader(target_input, target_fps=None):
        return ReaderFactory.create(target_input, target_fps)

        # return VideoReader
