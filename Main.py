#-*- coding: utf-8 -*-

import sys
import AudioResolutionApp
from PyQt5.QtWidgets import QApplication

# import moviepy.editor as mpe
import os

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AudioResolutionApp.AudioResolutionApp()
    sys.exit(app.exec_())

# import pyaudio
# p = pyaudio.PyAudio()
# for i in range(p.get_device_count()):
#     dev = p.get_device_info_by_index(i)
#     print((i, dev['name'], dev['maxInputChannels']))