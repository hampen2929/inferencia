from .reader_factory import ReaderFactory


class ReaderManager():
    def get_reader(target_input):
        return ReaderFactory.create(target_input)

        # return VideoReader
