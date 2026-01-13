import sys
import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
from psutil import virtual_memory
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
from PyQt5 import QtWidgets, uic, QtCore
import threading
import random
import subprocess
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from YouTubeDownload import YouTubeDownload
from WebDisplay import WebDisplay
from MusicTools import MusicSeparation

class MyApp(QtWidgets.QMainWindow):
    #初始化
    def __init__(self):
        super(MyApp, self).__init__()
        uic.loadUi("score/UI.ui", self)
        self.progressBar.setValue(0)

        self.searchButton.clicked.connect(self.start_separation)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.progress = 0

    #用彈窗提示錯誤
    @QtCore.pyqtSlot(str, str)
    def show_error_message(self, title, message):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

    #停止計時器
    @QtCore.pyqtSlot()
    def stop_timer(self):
        self.timer.stop()
    #開始時介面設置
    def start_separation(self):
        self.progress = 0
        self.progressBar.setValue(0)
        self.label_OUTPUT.setText("音源分離中...")

        self.timer.start(100)
        thread = threading.Thread(target=self.run_separation)
        thread.start()
    #隨機增加進度條
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
        #嘗試載入模型
        try:
            bundle = HDEMUCS_HIGH_MUSDB
            #bundle = DEMUCS_HTDEMOS
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
            #如果有NVIDIA顯卡且顯存足夠則使用cuda，否則用CPU
            GPUmem = torch.cuda.mem_get_info()[1]
            GPUmem = GPUmem /1024 / 1024 / 1024
            SYSmem = virtual_memory().total /1024 / 1024 / 1024 / 2
            if torch.cuda.is_available() and SYSmem + GPUmem > 16:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            model = model.to(device)

            #取得音樂檔案
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
            #如果檔案格式為wav
            if ext == '.wav':
                waveform, sample_rate = torchaudio.load(input_file)

            #否則ffmpeg轉檔
            else:
                temp_wav = "temp_convert.wav"
                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", input_file, "-acodec", "pcm_s16le", "-ar", "44100", temp_wav
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    #用soundfile讀temp wav
                    data, sample_rate = sf.read(temp_wav, dtype='float32')
                    if data.ndim == 1:
                        data = data[:, None]
                    waveform = torch.from_numpy(data.T)
                
                #如果ffmpeg轉檔失敗
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

            #重採樣
            if sample_rate != bundle.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
                waveform = resampler(waveform)

            waveform = waveform.to(device)
            
            #開始執行分離
            with torch.no_grad():
                try:
                    sources = model(waveform.unsqueeze(0))[0]
                #如果顯存不足改CPU跑
                except RuntimeError as oom_error:
                    if 'out of memory' in str(oom_error).lower() and device.type == 'cuda':
                        print("GPU 記憶體不足，切換到 CPU 重跑...")
                        torch.cuda.empty_cache()
                        device = torch.device('cpu')
                        model = model.to(device)
                        waveform = waveform.to(device)
                        sources = model(waveform.unsqueeze(0))[0]
                    else:
                        raise oom_error
            
            #取得輸出路徑及建立資料夾
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_dir = os.path.join(os.getcwd(), "score/output", base_name)
            os.makedirs(output_dir, exist_ok=True)

            #混音(Bass、Drum、Other)
            other_mix = sources[0] + sources[1] + sources[2]
            vocals = sources[3]
            
            #存檔
            torchaudio.save(os.path.join(output_dir, base_name + '-vocals.wav'), vocals.cpu(), bundle.sample_rate)
            torchaudio.save(os.path.join(output_dir, base_name + '-other.wav'), other_mix.cpu(), bundle.sample_rate)
            
            #輸出儲存位置
            print(f"已儲存：{os.path.join(output_dir, base_name + '-vocals.wav')}")
            print(f"已儲存：{os.path.join(output_dir, base_name + '-other.wav')}")
            
            #清空顯存及暫存資料
            del sources
            del waveform
            torch.cuda.empty_cache()
            
            #設定進度條及label
            QtCore.QMetaObject.invokeMethod(self, "stop_timer", QtCore.Qt.QueuedConnection)
            QtCore.QMetaObject.invokeMethod(
                self.progressBar,
                "setValue",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, 100)
            )
            self.update_output_label("音源分離完成！")
        
        #如果有錯誤
        except Exception as e:
            #嘗試刪除暫存音樂
            try:
                if sources is not None:
                    del sources
                if waveform is not None:
                    del waveform
            except:
                pass
                
            #清顯存
            torch.cuda.empty_cache()
            
            #顯示錯誤
            QtCore.QMetaObject.invokeMethod(self, "stop_timer", QtCore.Qt.QueuedConnection)
            QtCore.QMetaObject.invokeMethod(
                self,
                "show_error_message",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "執行錯誤"),
                QtCore.Q_ARG(str, str(e))
            )
            print(f"執行錯誤：{e}")
    
    #label輸出
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
