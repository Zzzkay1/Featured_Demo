import sys
import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
from PyQt5 import QtWidgets, uic, QtCore
import threading
import random
import subprocess
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        uic.loadUi("score/UI.ui", self)
        self.progressBar.setValue(0)

        self.searchButton.clicked.connect(self.start_separation)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.progress = 0

    @QtCore.pyqtSlot(str, str)
    def show_error_message(self, title, message):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

    @QtCore.pyqtSlot()
    def stop_timer(self):
        self.timer.stop()

    def start_separation(self):
        self.progress = 0
        self.progressBar.setValue(0)
        self.label_OUTPUT.setText("音源分離中...")

        self.timer.start(100)
        thread = threading.Thread(target=self.run_separation)
        thread.start()

    def update_progress(self):
        if self.progress < 95:
            increment = random.randint(1, 3)
            self.progress = min(self.progress + increment, 95)
            self.progressBar.setValue(self.progress)
            next_interval = random.randint(50, 300)
            self.timer.setInterval(next_interval)

    def run_separation(self):
        sources = None
        waveform = None
        try:
            bundle = HDEMUCS_HIGH_MUSDB
            model = bundle.get_model()
        except Exception as e:
            QtCore.QMetaObject.invokeMethod(self, "stop_timer", QtCore.Qt.QueuedConnection)
            QtCore.QMetaObject.invokeMethod(
                self,
                "show_error_message",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "模型載入錯誤"),
                QtCore.Q_ARG(str, str(e))
            )
            return

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            input_file = 'score/music/' + self.lineEdit.text()
            if not os.path.exists(input_file):
                self.update_output_label(f"找不到輸入檔案：{input_file}")
                QtCore.QMetaObject.invokeMethod(
                    self,
                    "show_error_message",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, "執行錯誤"),
                    QtCore.Q_ARG(str, f"找不到輸入檔案: {input_file}")
                )
                QtCore.QMetaObject.invokeMethod(self, "stop_timer", QtCore.Qt.QueuedConnection)
                return

            ext = os.path.splitext(input_file)[1].lower()

            if ext == '.wav':
                waveform, sample_rate = torchaudio.load(input_file)

            else:
                temp_wav = "temp_convert.wav"
                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", input_file, "-acodec", "pcm_s16le", "-ar", "44100", temp_wav
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # 用 soundfile 讀 temp wav
                    data, sample_rate = sf.read(temp_wav, dtype='float32')
                    if data.ndim == 1:
                        data = data[:, None]
                    waveform = torch.from_numpy(data.T)

                except subprocess.CalledProcessError as ffmpeg_err:
                    QtCore.QMetaObject.invokeMethod(self, "stop_timer", QtCore.Qt.QueuedConnection)
                    QtCore.QMetaObject.invokeMethod(
                        self,
                        "show_error_message",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, "ffmpeg 轉檔失敗"),
                        QtCore.Q_ARG(str, str(ffmpeg_err))
                    )
                    return
                finally:
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)

            # 重採樣
            if sample_rate != bundle.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
                waveform = resampler(waveform)

            waveform = waveform.to(device)

            with torch.no_grad():
                sources = model(waveform.unsqueeze(0))[0]

            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_dir = os.path.join(os.getcwd(), "score/output", base_name)
            os.makedirs(output_dir, exist_ok=True)

            other_mix = sources[0] + sources[1] + sources[2]
            vocals = sources[3]

            torchaudio.save(os.path.join(output_dir, base_name + '-vocals.wav'), vocals.cpu(), bundle.sample_rate)
            torchaudio.save(os.path.join(output_dir, base_name + '-other.wav'), other_mix.cpu(), bundle.sample_rate)

            print(f"已儲存：{os.path.join(output_dir, base_name + '-vocals.wav')}")
            print(f"已儲存：{os.path.join(output_dir, base_name + '-other.wav')}")

            del sources
            del waveform
            torch.cuda.empty_cache()

            QtCore.QMetaObject.invokeMethod(self, "stop_timer", QtCore.Qt.QueuedConnection)
            QtCore.QMetaObject.invokeMethod(
                self.progressBar,
                "setValue",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, 100)
            )
            self.update_output_label("音源分離完成！")

        except Exception as e:
            try:
                if sources is not None:
                    del sources
                if waveform is not None:
                    del waveform
            except:
                pass
            torch.cuda.empty_cache()

            QtCore.QMetaObject.invokeMethod(self, "stop_timer", QtCore.Qt.QueuedConnection)
            QtCore.QMetaObject.invokeMethod(
                self,
                "show_error_message",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "執行錯誤"),
                QtCore.Q_ARG(str, str(e))
            )
            print(f"執行錯誤：{e}")

    def update_output_label(self, text):
        QtCore.QMetaObject.invokeMethod(
            self.label_OUTPUT,
            "setText",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, text)
        )

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
