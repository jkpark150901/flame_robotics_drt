'''
CSV(Comma Separated Value) Recorder Class
@author Byunghun Hwang<bh.hwang@iae.re.kr>
'''

import csv
import typing
import pathlib
from PyQt6.QtCore import QObject
from util.logger.console import ConsoleLogger

class CSVRecorder(QObject):
    def __init__(self, dirpath:pathlib.Path, filename:str) -> None:
        super().__init__()
        
        self.__console = ConsoleLogger.get_logger()
        self.__save_path = dirpath / f"{filename}.csv"
        self.__writer = None
        self.__csv_file = None
        self.__is_working = False
        
    # start write
    def start(self):
        self.__csv_file = open(self.__save_path.as_posix(), mode="a+", newline='')
        self.__writer = csv.writer(self.__csv_file)
        self.__is_working = True
    
    # stop write
    def stop(self):
        self.__is_working = False
    
    # write row in csv file
    def write_row(self, data:list):
        if self.__is_working:
            self.__writer.writerow(data)