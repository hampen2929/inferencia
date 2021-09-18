# -*- coding: UTF-8 -*-
import inspect
import os
import os.path as osp
from logging import (DEBUG, INFO, WARN, Formatter, StreamHandler, getLogger,
                     handlers)


class Logger:
    def __init__(
        self,
        name=__name__,
        logger_level=INFO,  # DEBUG
        handler_level=INFO,
        sthandler_level=INFO,  # DEBUG
    ):
        # ロガー生成
        self.logger = getLogger(name)
        self.logger.setLevel(logger_level)
        self.logger.propagate = False
        formatter = Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)7s %(message)s [%(name)s  %(processName)s - %(threadName)s]",
            datefmt="%Y/%m/%d %H:%M:%S",
        )

        # 時刻ローテーション
        file_path = "log/{}.log".format(__name__)
        os.makedirs(osp.dirname(file_path), exist_ok=True)
        if not self.logger.hasHandlers():
            handler = handlers.TimedRotatingFileHandler(
                filename=file_path, encoding="UTF-8", when="D", backupCount=7
            )
            # サイズローテーション
            """
            handler = handlers.RotatingFileHandler(filename='/var/log/inferencia_package/{}.log'.format(__name__),
                                                encoding='UTF-8',
                                                maxBytes=1048576,
                                                backupCount=3)
            """
            # ログファイル設定
            handler.setLevel(handler_level)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            # 標準出力用 設定： DEBUG レベルまで標準出力する
            sthandler = StreamHandler()
            sthandler.setLevel(sthandler_level)
            sthandler.setFormatter(formatter)
            self.logger.addHandler(sthandler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warn(self, msg):
        self.logger.warning(msg)

    def error(self, msg, exc_info=False):
        self.logger.error(msg, exc_info=exc_info)

    def critical(self, msg):
        self.logger.critical(msg)
