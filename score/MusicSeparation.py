import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
from psutil import virtual_memory
import subprocess
import os
import soundfile as sf
import tkinter as tk
from torchaudio.transforms import Fade

class MusicSeparation:
    def __init__(self):
        self.sources = None
        self.waveform = None

    @staticmethod
    def _separate_sources(model, mix, sample_rate, device, segment=15.0, overlap=0.1):
        """
        分段執行音源分離，避免一次丟整首造成爆顯存
        """
        batch, channels, length = mix.shape

        chunk_len = int(sample_rate * segment * (1 + overlap))
        start = 0
        end = chunk_len
        overlap_frames = int(overlap * sample_rate)
        fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

        final = torch.zeros(batch, len(model.sources), channels, length, device=device)

        while start < length - overlap_frames:
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out

            if start == 0:
                fade.fade_in_len = overlap_frames
                start += chunk_len - overlap_frames
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0

        return final

    @staticmethod
    def run_separation(audio_path, segment = 15.0, use_cuda = True):
        #嘗試載入模型
        try:
            bundle = HDEMUCS_HIGH_MUSDB
            model = bundle.get_model()
        except Exception as e:
            return e

        #如果有NVIDIA顯卡且顯存足夠則使用cuda，否則用CPU
        try:
            if torch.cuda.is_available():
                GPUmem = torch.cuda.mem_get_info()[1] / 1024 / 1024 / 1024
                if GPUmem < 1:
                    segment = 5
                elif GPUmem <= 2:
                    segment = 8
                elif GPUmem <= 4:
                    segment = 12
            SYSmem = virtual_memory().total / 1024 / 1024 / 1024 / 2
            if torch.cuda.is_available() and use_cuda:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            model = model.to(device)

            

            #取得音樂檔案
            input_file = audio_path.strip('"')
            if not os.path.exists(input_file):
                return f"找不到輸入檔案：{input_file}"

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
                    return ffmpeg_err
                #完成刪檔案
                finally:
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)

            #重採樣
            if sample_rate != bundle.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
                waveform = resampler(waveform)
                sample_rate = bundle.sample_rate

            waveform = waveform.to(device)

            #使用分段方式做分離(overlap=0.1)
            sources = MusicSeparation._separate_sources(
                model=model,
                mix=waveform.unsqueeze(0),
                segment = segment,
                sample_rate=sample_rate,
                device=device
            )[0]

            #取得輸出路徑及建立資料夾
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_dir = os.path.join(os.getcwd(), "score/output", base_name)
            os.makedirs(output_dir, exist_ok=True)

            #混音(Bass、Drum、Other)
            other_mix = sources[0] + sources[1] + sources[2]
            vocals = sources[3]

            #音檔儲存
            torchaudio.save(os.path.join(output_dir, base_name + '-vocals.wav'), vocals.cpu(), sample_rate)
            torchaudio.save(os.path.join(output_dir, base_name + '-other.wav'), other_mix.cpu(), sample_rate)

            #輸出儲存位置
            print(f"已儲存：{os.path.join(output_dir, base_name + '-vocals.wav')}")
            print(f"已儲存：{os.path.join(output_dir, base_name + '-other.wav')}")

            #清空顯存及暫存資料
            del sources
            del waveform
            torch.cuda.empty_cache()

            output = [
                f"{os.path.join(output_dir, base_name + '-other.wav')}",
                f"{os.path.join(output_dir, base_name + '-vocals.wav')}"
            ]
            return output

        #如果有錯誤
        except Exception as e:
            #嘗試刪除暫存音樂
            try:
                if 'sources' in locals():
                    del sources
                if 'waveform' in locals():
                    del waveform
            except:
                pass
            #清顯存
            torch.cuda.empty_cache()
            return e

        
    #ASR辨識
    #---ASR辨識---
    @staticmethod
    def run_ASR(audio="" ,Input_language = ""):
        import whisper
        from tkinter import filedialog
        
        #建立Tk主視窗，但隱藏
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        #傳入值如果為空由使用者手動選擇音訊檔案
        if audio == "":
            current_dir = os.getcwd()
            file_path = filedialog.askopenfilename(
                title="請選擇音訊檔案",
                initialdir=current_dir,
                filetypes=[("音訊檔案", "*.wav *.mp3 *.m4a *.flac"), ("所有檔案", "*.*")]
            )
            if not file_path:
                print("未選擇檔案")
                exit()
        else:
            file_path = audio.strip('"')

        file_path = os.path.abspath(file_path)
        print("選擇的檔案:", file_path)

        #載入Whisper模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("large-v2", device=device)

        #轉檔為16k單聲道wav
        temp_wav = "temp_for_asr.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", file_path, "-ac", "1", "-ar", "16000", "-f", "wav", temp_wav
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        #轉錄音訊
        if(Input_language == ""):
            result = model.transcribe(temp_wav, beam_size=5, word_timestamps=True ,temperature=0)
        else:
            result = model.transcribe(temp_wav, beam_size=5, word_timestamps=True ,temperature=0 ,language=Input_language)
        os.remove(temp_wav)
        print("偵測語言:", result["language"])

        #取得音訊檔名稱(不含副檔名)
        audio_dir = os.path.dirname(file_path)
        filename = os.path.splitext(os.path.basename(file_path))[0]

        #---自訂拆段，每段不超過10秒---
        max_segment_len = 5
        segments_for_output = []
        current_seg = {"start": None, "end": None, "text": ""}

        for seg in result["segments"]:
            if current_seg["start"] is None:
                current_seg["start"] = seg["start"]
            current_seg["text"] += seg["text"].strip() + " "
            current_seg["end"] = seg["end"]

            #超過10秒或段落內有標點就拆段
            if (current_seg["end"] - current_seg["start"] >= max_segment_len) or any(p in seg["text"] for p in [".", "?", "!", ","]):
                segments_for_output.append(current_seg.copy())
                current_seg = {"start": None, "end": None, "text": ""}

        #最後一段
        if current_seg["text"].strip():
            segments_for_output.append(current_seg)

        #---SRT---
        with open(os.path.join(audio_dir, f"{filename}.srt"), "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments_for_output, start=1):
                text = seg['text'].strip()
                if not text:
                    continue
                f.write(f"{i}\n")
                start_time = seg["start"]
                end_time = seg["end"]
                start_time_str = f"{int(start_time//3600):02}:{int((start_time%3600)//60):02}:{int(start_time%60):02},{int((start_time%1)*1000):03}"
                end_time_str = f"{int(end_time//3600):02}:{int((end_time%3600)//60):02}:{int(end_time%60):02},{int((end_time%1)*1000):03}"
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{text}\n\n")

        #---VTT---
        with open(os.path.join(audio_dir, f"{filename}.vtt"), "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in segments_for_output:
                text = seg['text'].strip()
                if not text:
                    continue
                start_time = seg["start"]
                end_time = seg["end"]
                start_time_str = f"{int(start_time//3600):02}:{int((start_time%3600)//60):02}:{int(start_time%60):02}.{int((start_time%1)*1000):03}"
                end_time_str = f"{int(end_time//3600):02}:{int((end_time%3600)//60):02}:{int(end_time%60):02}.{int((end_time%1)*1000):03}"
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{text}\n\n")

        #---TXT---
        with open(os.path.join(audio_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
            for seg in segments_for_output:
                text = seg['text'].strip()
                if not text:
                    continue
                start_time = seg["start"]
                end_time = seg["end"]
                start_time_str = f"{int(start_time//3600):02}:{int((start_time%3600)//60):02}:{int(start_time%60):02}.{int((start_time%1)*1000):03}"
                end_time_str = f"{int(end_time//3600):02}:{int((end_time%3600)//60):02}:{int(end_time%60):02}.{int((end_time%1)*1000):03}"
                f.write(f"[{start_time_str} --> {end_time_str}] {text}\n")

        print(f"已輸出至資料夾:{audio_dir}")
        torch.cuda.empty_cache()
        del model
        return [os.path.join(audio_dir, f"{filename}.srt"),result["language"]]
