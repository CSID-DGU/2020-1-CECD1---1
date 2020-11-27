import pyaudio
import wave

class AudioRecorder:
    def __init__(self):
        self.CHUNK = 1024  # Record in chunks of 1024 samples
        self.sample_format = pyaudio.paInt16  # 16 bits per sample
        self.channels = 1
        self.is_recording = False
        self.fs = 44100  # Record at 44,100 samples per second
        self.record_second = 10
        self.file_name = 'audiorecord.wav'
        self.p = pyaudio.PyAudio() # Create an interface to PortAuido
        self.inputDevice = 1
    def audio_record(self):

        stream = self.p.open(format=self.sample_format,
                        channels=self.channels,
                        rate=self.fs,
                        frames_per_buffer=self.CHUNK,
                        input_device_index=self.inputDevice,
                        input=True)
        
        frames = []  # initialize array to store frames


        while self.is_recording is True:
            data = stream.read(self.CHUNK)
            frames.append(data)

        # Store data in chunks for record_second seconds
        # for i in range(0, int(self.fs / self.CHUNK * self.record_second)):
        #     data = stream.read(self.CHUNK)
        #     frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Terminate the PortAudio interface
        self.p.terminate()

        # Save the recorded data as a WAV file
        wf = wave.open(self.file_name, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(frames))
        wf.close()

    # def getaudiodevices(self):
    #     # for i in range(self.p.get_device_count()):
    #     #     dev = self.p.get_device_info_by_index(i)
    #     #     print((i, dev['name'], dev['maxInputChannels']))
    #     a.desc = a.cb.currentIndex() + 1
    #     a.INDEX_OF_CHOSEN_INPUT_DEVICE = a.desc
    #     print(a.desc)
