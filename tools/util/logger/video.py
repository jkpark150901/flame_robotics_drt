'''
Video Recorder Class
@author Byunghun Hwang<bh.hwang@iae.re.kr>
'''

import cv2
try:
    from PyQt6.QtCore import QObject, pyqtSignal
    from PyQt6.QtGui import QImage
except ImportError:
    from PyQt5.QtCore import QObject, pyqtSignal
    from PyQt5.QtGui import QImage
    
import pathlib
from abc import *
from util.logger.console import ConsoleLogger
from datetime import datetime
import numpy as np


class VideoRecorder(QObject):
    def __init__(self, dirpath:pathlib.Path, filename:str, resolution:(int,int), fps:float, ext:str="avi"):
        super().__init__()
        
        self.__console = ConsoleLogger.get_logger()
        
        self.__writer = None
        self.__is_recording = False
        self.__dirpath = dirpath
        self.__filename = filename
        self.__resolution = resolution
        self.__fps = fps
        self.__ext = ext
        self.__fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        self.__video_outfile_absolute = None

    def start(self):
        if self.__is_recording:
            self.__console.warning("Video recording is now in progress...")
            return
        
        # create directory named from date
        record_start_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        video_out_path = self.__dirpath / record_start_datetime
        video_out_path.mkdir(parents=True, exist_ok=True)
        
        self.__video_outfile_absolute = (video_out_path / f"{self.__filename}.{self.__ext}")
        self.__console.info(f"Recording in {self.__video_outfile_absolute.as_posix()}")
        
        # create video writer
        self.__writer = cv2.VideoWriter(self.__video_outfile_absolute.as_posix(), 
                                        self.__fourcc,
                                        self.__fps,
                                        self.__resolution)
        
        self.__is_recording = True
    
    def pause(self):
        self.__console.warning("Not support yet.")
    
    def stop(self):
        print("recoring stop")
        if self.__writer:
            self.__writer.release()
        self.__is_recording = False
    
    # write a frame
    def write_frame(self, image:np.ndarray, fps:float):
        if self.__is_recording:
            self.__writer.write(image)