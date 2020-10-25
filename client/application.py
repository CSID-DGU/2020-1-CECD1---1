import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
from PIL import ImageGrab
import datetime
import time
import pyaudio
import wave
import threading
import ffmpeg

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(BASE_DIR, "server")
sys.path.append(SERVER_DIR)
import server


class Recorder:
    def __init__(self):
        # 공통
        self.filename = "output.avi"
        self.recording = True
        self.fps = 24.0
        self.frame_counts = 1

        # 비디오
        self.video_filename = "video.avi"
        self.res = "720p"
        self.STD_DIMENSIONS = {
            "480p": (640, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160),
        }
        self.VIDEO_TYPE = {
            "avi": cv2.VideoWriter_fourcc(*"XVID"),
            "mp4": cv2.VideoWriter_fourcc(*"XVID"),
        }

        # 오디오
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 2
        self.format = pyaudio.paInt16
        self.audio_filename = "audio.wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
        )
        self.audio_frames = []

    # 비디오
    def change_res(self, cap, width, height):
        cap.set(3, width)
        cap.set(4, height)

    def get_dims(self, cap, res="1080p"):
        width, height = self.STD_DIMENSIONS["480p"]
        if res in self.STD_DIMENSIONS:
            width, height = self.STD_DIMENSIONS[res]

        self.change_res(cap, width, height)
        return width, height

    def get_video_type(self, filename):
        filename, ext = os.path.splitext(filename)
        if ext in self.VIDEO_TYPE:
            return self.VIDEO_TYPE[ext]
        return self.VIDEO_TYPE["avi"]

    def record_video(self):
        # cap = cv2.VideoCapture(0)
        # out = cv2.VideoWriter(
        #     self.filename,
        #     self.get_video_type(self.filename),
        #     25,
        #     self.get_dims(cap, self.res),
        # )
        out = cv2.VideoWriter(
            self.filename, self.get_video_type(self.filename), 25, (500, 490),
        )

        self.recording = True

        while self.recording:
            # ret, frame = cap.read()
            # if ret:
            #     out.write(frame)
            #     # cv2.imshow("frame", frame)
            #     self.frame_counts += 1
            #     time.sleep(1 / self.fps)

            img = ImageGrab.grab(bbox=(100, 10, 600, 500))
            img_np = np.array(img)
            # frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            out.write(img_np)
            # cv2.imshow("frame", img_np)

        # cap.release()
        out.release()
        cv2.destroyAllWindows()

    def record_audio(self):
        self.stream.start_stream()
        self.recording = True

        while self.recording:
            data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
            self.audio_frames.append(data)

    # 오디오/비디오 합치기
    def merge(self):
        original_audio = "audio.wav"
        server.audio_resolution("audio.wav")
        input_video = ffmpeg.input("./video.avi")
        input_audio = ffmpeg.input("./audio.wav..pr.wav")
        ffmpeg.concat(input_video, input_audio, v=1, a=1).output(
            "./result.avi"
        ).overwrite_output().run()

    # 녹화 제어
    def start(self):
        video_thread = threading.Thread(target=self.record_video)
        video_thread.start()
        # audio_thread = threading.Thread(target=self.record_audio)
        # audio_thread.start()

    def stop(self):
        self.recording = False

        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        waveFile = wave.open(self.audio_filename, "wb")
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.format))
        waveFile.setframerate(self.rate)
        waveFile.writeframes(b"".join(self.audio_frames))
        waveFile.close()
        self.merge()


class AudioSuperResolutionRecorderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.record_file_list = []
        self.recorder = Recorder()
        # self.recorder.merge()
        self.set_up_ui()

    def set_up_ui(self):
        self.setWindowTitle("Audio Super Resolution Recorder")
        self.setFixedSize(800, 600)

        root_layout = QVBoxLayout()

        # record button 생성 및 이벤트 리스너 등록
        self.record_btn = self.record_button(btn_text="record", maximum_width=130)
        self.record_btn.clicked.connect(self.record_event_listener)

        # stop button 생성 및 이벤트 리스너 등록
        self.stop_btn = self.stop_button(btn_text="stop", maximum_width=130)
        self.stop_btn.clicked.connect(self.stop_event_listener)

        root_layout.addWidget(self.record_btn)
        root_layout.addWidget(self.stop_btn)

        self.setLayout(root_layout)

    def record_button(self, btn_text, maximum_width=100, maximum_height=50):
        button = QPushButton(btn_text)
        button.setMaximumSize(maximum_width, maximum_height)
        button.setLayoutDirection(Qt.RightToLeft)
        return button

    def stop_button(self, btn_text, maximum_width=100, maximum_height=50):
        button = QPushButton(btn_text)
        button.setMaximumSize(maximum_width, maximum_height)
        button.setLayoutDirection(Qt.RightToLeft)
        return button

    def file_select_event_listener(self):
        file_name = QFileDialog.getOpenFileName(self)
        self.record_file_list.append(self.get_file_dict(file_name[0]))
        len_of_list = len(self.record_file_list)
        self.add_table_item(len_of_list - 1, self.record_file_list[len_of_list - 1])

    def create_table_widget(self):
        # TableWidget 생성 및 스타일시트 적용
        file_list_table = QTableWidget(self)
        file_list_table.setStyleSheet("background-color:#3b3b3b; color: white")
        # 헤더 숨기기
        column_header = file_list_table.horizontalHeader()
        column_header.hide()

        # row count 숨기기
        row_header = file_list_table.verticalHeader()
        row_header.hide()

        # column size 조정
        file_list_table.setRowCount(0)
        file_list_table.setColumnCount(5)
        file_list_table.setColumnWidth(0, 350)
        file_list_table.setColumnWidth(1, 100)
        file_list_table.setColumnWidth(2, 100)
        file_list_table.setColumnWidth(3, 100)
        file_list_table.setColumnWidth(4, 100)

        return file_list_table

    # file_name을 받아서 해당 파일의 정보를 딕셔너리로 반환
    def get_file_dict(self, file_name):
        file_size = round(
            os.stat(file_name).st_size / (10 ** 6), 2
        )  # bytes to megabytes
        file_dict = {
            "name": file_name,
            "size": str(file_size) + " MB",
            "record_rate": 0,
        }
        return file_dict

    def add_table_item(self, row_cnt, file_info):
        download_btn = QPushButton("download")
        download_btn.clicked.connect(self.download_button_click_listener)

        delete_btn = QPushButton("delete")
        delete_btn.clicked.connect(self.delete_button_click_listener)

        self.file_list_table.insertRow(row_cnt)
        self.file_list_table.setItem(
            row_cnt, 0, QTableWidgetItem(str(file_info["name"]))
        )
        self.file_list_table.setItem(
            row_cnt, 1, QTableWidgetItem(str(file_info["size"]))
        )
        self.file_list_table.setItem(
            row_cnt, 2, QTableWidgetItem(str(file_info["record_rate"]) + "%")
        )
        self.file_list_table.setCellWidget(row_cnt, 3, download_btn)
        self.file_list_table.setCellWidget(row_cnt, 4, delete_btn)

    # download 버튼 클릭 이벤트 리스너
    def download_button_click_listener(self):
        index = self.file_list_table.selectedIndexes()
        for idx in index:
            row = idx.row()
            # text = self.file_list_table.item(row, 0).text()
            text = self.record_file_list[row]["name"]
            print(text)

    def delete_button_click_listener(self):
        # self.file_list_table.setSelectionBehavior(QAbstractItemView.SelectRows) # row 단위로 선택 가능
        index = self.file_list_table.selectedIndexes()
        for idx in index:
            row = idx.row()
            print(row)
            self.file_list_table.removeRow(row)
            del self.record_file_list[row]

        for item in self.record_file_list:
            print(item["name"])

    def record_event_listener(self):
        self.recorder.start()

    def stop_event_listener(self):
        self.recorder.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    application = AudioSuperResolutionRecorderApp()
    application.show()
    sys.exit(app.exec_())
