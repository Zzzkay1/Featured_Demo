import os
import torch
import torchaudio
import subprocess
import gc
import soundfile as sf
import tkinter as tk
from tkinter import filedialog
from psutil import virtual_memory
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
from torchaudio.transforms import Fade
import time


#將 transformers 移到全域引用,解決 Worker Error
try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
except ImportError:
    print("錯誤：找不到 transformers 套件。請執行 pip install transformers accelerate")
    raise

#設定 Torchaudio 後端,解決 UserWarning (針對 Windows 環境)
if os.name == 'nt':
    try:
        torchaudio.set_audio_backend("soundfile")
    except:
        pass #新版 torchaudio 可能不支援此語法,則依賴預設

class MusicTools:
    def __init__(self):
        self.sources = None
        self.waveform = None

    @staticmethod
    def _separate_sources(model, mix, sample_rate, device, segment=15.0, overlap=0.1):
        """
        分段執行音源分離,避免一次丟整首造成爆顯存
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
    def run_separation(audio_path, segment = 15.0, use_cuda = True) -> list|None:
        # 限制 PyTorch 的 CPU 執行緒數，避免背景任務佔滿 CPU 導致 Streamlit 網頁凍結
        if torch.get_num_threads() > 2:
            torch.set_num_threads(max(2, os.cpu_count() // 2 - 1))
        #嘗試載入模型
        try:
            bundle = HDEMUCS_HIGH_MUSDB
            model = bundle.get_model()
        except Exception as e:
            raise e

        #如果有NVIDIA顯卡且顯存足夠則使用cuda,否則用CPU
        try:
            if torch.cuda.is_available():
                GPUmem = torch.cuda.mem_get_info()[1] / 1024 / 1024 / 1024
                if GPUmem < 1:
                    segment = 5
                elif GPUmem <= 2:
                    segment = 8
                elif GPUmem <= 4:
                    segment = 12
            
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
                #修正 warning: 明確指定 backend
                waveform, sample_rate = torchaudio.load(input_file, backend="soundfile")
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
                        try:
                            os.remove(temp_wav)
                        except:
                            pass

            #重採樣
            if sample_rate != bundle.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
                waveform = resampler(waveform)
                sample_rate = bundle.sample_rate

            waveform = waveform.to(device)

            #使用分段方式做分離(overlap=0.1)
            sources = MusicTools._separate_sources(
                model=model,
                mix=waveform.unsqueeze(0),
                segment = segment,
                sample_rate=sample_rate,
                device=device
            )[0]

            #取得輸出路徑及建立資料夾
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_dir = os.path.join(os.getcwd(), "source/output", base_name)
            os.makedirs(output_dir, exist_ok=True)

            #混音(Bass、Drum、Other)
            other_mix = sources[0] + sources[1] + sources[2]
            vocals = sources[3]

            #音檔儲存 (指定 backend 消除警告)
            torchaudio.save(os.path.join(output_dir, base_name + '-vocals.wav'), vocals.cpu(), sample_rate, backend="soundfile")
            torchaudio.save(os.path.join(output_dir, base_name + '-other.wav'), other_mix.cpu(), sample_rate, backend="soundfile")

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
    @staticmethod
    def run_ASR(audio="", Input_language=""):
        import torch
        import gc
        import os
        import math
        import soundfile as sf
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        
        # 限制 PyTorch 的 CPU 執行緒數，避免背景任務佔滿 CPU 導致 Streamlit 網頁凍結
        if torch.get_num_threads() > 2:
            torch.set_num_threads(max(2, os.cpu_count() // 2 - 1))

        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
        except:
            pass 

        if audio == "":
            current_dir = os.getcwd()
            file_path = filedialog.askopenfilename(
                title="請選擇音訊檔案",
                initialdir=current_dir,
                filetypes=[("音訊檔案", "*.wav *.mp3 *.m4a *.flac"), ("所有檔案", "*.*")]
            )
            if not file_path:
                print("未選擇檔案")
                return None
        else:
            file_path = audio.strip('"')

        file_path = os.path.abspath(file_path)
        audio_dir = os.path.dirname(file_path)
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        #獲取音訊總長度
        try:
            info = sf.info(file_path)
            audio_duration = info.duration
        except:
            audio_duration = 999999.0

        #--- 載入模型 ---
        if Input_language == "zh" or Input_language == "" or Input_language == "中文":
            tempmodel = "Zzzkay1/Whisper-medium-zh" 
        if Input_language == "en" or Input_language == "英文":
            tempmodel = "Zzzkay1/Whisper-medium-en" 
        if Input_language == "nan" or Input_language == "閩南語":
            tempmodel = "Zzzkay1/Whisper-medium-nan"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"正在載入模型 {tempmodel}...")
        try:
            processor = AutoProcessor.from_pretrained(tempmodel)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                tempmodel, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)
        except Exception as e:
            print(f"模型載入失敗: {e}")
            return None

        #--- 音量 VAD預處理與過濾 ---
        print("正在進行音量 VAD過濾...")
        processed_file_path = os.path.join(audio_dir, f"{filename}_vad_processed.wav")
        
        # 預設為整首保留，避免 try 區塊失敗時後續找不到 speech_timestamps
        speech_timestamps = [] 
        
        try:
            # 改用 torchaudio 實作讀寫，避免 torch.hub 需要網路連線或快取
            def read_audio(path, sampling_rate=16000):
                wav, sr = torchaudio.load(path, backend="soundfile")
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)  # 轉單聲道
                if sr != sampling_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
                    wav = resampler(wav)
                return wav.squeeze(0)  # 回傳 1D tensor

            def save_audio(path, tensor, sampling_rate=16000):
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                torchaudio.save(path, tensor, sampling_rate, backend="soundfile")
            
            #讀取原始音訊
            wav = read_audio(file_path, sampling_rate=16000)
            
            # --- 新增：音量正規化 (Peak Normalization) ---
            # 將音檔的最大音量拉到 1.0 (0 dB)，確保每首歌的音量基準一致
            max_amp = torch.max(torch.abs(wav))
            if max_amp > 0:
                wav = wav / max_amp
            
            # --- 定義純粹靠音量的 VAD 函數 ---
            def get_energy_timestamps(wav_tensor, sr=16000, threshold_db=-35.0, window_ms=50, min_silence_ms=2500, pad_ms=500):
                window_samples = int(sr * (window_ms / 1000.0))
                if len(wav_tensor) < window_samples:
                    return [{'start': 0, 'end': len(wav_tensor)}]
                    
                # 計算每小段 (預設50ms) 的 RMS 音量
                unfolded = wav_tensor.unfold(0, window_samples, window_samples)
                rms = torch.sqrt(torch.mean(unfolded ** 2, dim=1))
                
                # 分貝轉振幅門檻
                rms_threshold = 10 ** (threshold_db / 20.0)
                active_windows = rms > rms_threshold
                
                # 將符合音量的小片段轉換成 start, end 時間點
                timestamps = []
                in_speech = False
                start_sample = 0
                for i, active in enumerate(active_windows):
                    if active and not in_speech:
                        in_speech = True
                        start_sample = i * window_samples
                    elif not active and in_speech:
                        in_speech = False
                        timestamps.append({'start': start_sample, 'end': i * window_samples})
                if in_speech:
                    timestamps.append({'start': start_sample, 'end': len(wav_tensor)})
                    
                # 1. 合併短暫靜音
                min_silence_samples = int(sr * (min_silence_ms / 1000.0))
                merged = []
                for s in timestamps:
                    if not merged:
                        merged.append(s)
                    elif s['start'] - merged[-1]['end'] < min_silence_samples:
                        merged[-1]['end'] = s['end']
                    else:
                        merged.append(s)
                        
                # 2. 加上前後緩衝 (保護尾音與殘響)
                pad_samples = int(sr * (pad_ms / 1000.0))
                final_stamps = []
                for s in merged:
                    start = max(0, s['start'] - pad_samples)
                    end = min(len(wav_tensor), s['end'] + pad_samples)
                    # 若加上緩衝後與前一段重疊，則合併
                    if final_stamps and final_stamps[-1]['end'] >= start:
                        final_stamps[-1]['end'] = max(final_stamps[-1]['end'], end)
                    else:
                        final_stamps.append({'start': int(start), 'end': int(end)})
                return final_stamps
            # ----------------------------

            # 執行音量 VAD
            speech_timestamps = get_energy_timestamps(
                wav, 
                sr=16000, 
                threshold_db=-28.0,   # 稍微提高門檻(-28dB)，讓微小的呼吸聲被視為靜音
                min_silence_ms=300,  # 停頓超過 0.3 秒就切斷 (斷句更細緻)
                pad_ms=100            # 前後僅保留 0.1 秒緩衝 (避免緩衝重疊導致兩句黏合)
            )

            if len(speech_timestamps) > 0:
                vad_audio = torch.zeros_like(wav)
                
                #複製清洗後的片段
                for stamp in speech_timestamps:
                    vad_audio[stamp['start']:stamp['end']] = wav[stamp['start']:stamp['end']]
                
                #儲存
                save_audio(processed_file_path, vad_audio, sampling_rate=16000)
                print(f"音量過濾完成,暫存檔: {processed_file_path}")
                
                wav = vad_audio 
            else:
                print("警告: 整首都是靜音，將使用原始檔案。")
                speech_timestamps = [{'start': 0, 'end': len(wav)}]

        except Exception as e:
            print(f"VAD 處理失敗,將使用原始檔案: {e}")
            #如果失敗,wav 保持原樣,繼續執行
            if 'wav' in locals():
                speech_timestamps = [{'start': 0, 'end': len(wav)}]

        #--- 微切分與辨識 ---
        segments_for_output = []
        print(f"準備對 {len(speech_timestamps)} 個片段進行長度檢查與辨識...")

        #強制切分上限 (秒)
        MAX_CHUNK_DURATION = 25.0 

        #準備要處理的片段列表
        final_chunks_to_process = []
        for stamp in speech_timestamps:
            start_sample = stamp['start']
            end_sample = stamp['end']
            duration_sec = (end_sample - start_sample) / 16000

            if duration_sec <= MAX_CHUNK_DURATION:
                final_chunks_to_process.append((start_sample, end_sample))
            else:
                #如果單一句子超過 25 秒,強制切斷
                num_sub_chunks = math.ceil(duration_sec / MAX_CHUNK_DURATION)
                chunk_samples = int(MAX_CHUNK_DURATION * 16000)
                for i in range(num_sub_chunks):
                    sub_start = start_sample + (i * chunk_samples)
                    sub_end = min(sub_start + chunk_samples, end_sample)
                    if (sub_end - sub_start) / 16000 < 0.2: continue
                    final_chunks_to_process.append((sub_start, sub_end))

        #設定解碼參數
        #使用Input_language參數,若為 None 或 "auto" 則自動偵測
        lang = Input_language if Input_language and Input_language.lower() != "auto" else None
        #forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")

        for idx, (start_sample, end_sample) in enumerate(final_chunks_to_process):
            
            vad_start_time = start_sample / 16000
            vad_end_time = end_sample / 16000
            
            #這裡取出的segment_wav是經過VAD過濾的音訊
            segment_wav = wav[start_sample:end_sample].numpy()
            
            try:
                #轉特徵
                input_features = processor(
                    segment_wav, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(device).to(torch_dtype)

                #直接生成
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_features,
                        #forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255, 
                        temperature=0.0,
                        repetition_penalty=1.2
                    )

                #解碼
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            except Exception as e:
                print(f"片段錯誤 ({vad_start_time:.1f}s): {e}")
                continue

            if len(text) < 1: continue
            if segments_for_output and text == segments_for_output[-1]['text']: continue

            segments_for_output.append({
                "start": vad_start_time,
                "end": vad_end_time,
                "text": text
            })

            if idx % 5 == 0:
                print(f"進度 {idx}/{len(final_chunks_to_process)}: {text[:10]}...")

        #--- 輸出 SRT ---
        def format_time_srt(time_sec):
            hours = int(time_sec // 3600)
            minutes = int((time_sec % 3600) // 60)
            seconds = int(time_sec % 60)
            milliseconds = int((time_sec % 1) * 1000)
            return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
            
        srt_path = os.path.join(audio_dir, f"{filename}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments_for_output, start=1):
                f.write(f"{i}\n")
                f.write(f"{format_time_srt(seg['start'])} --> {format_time_srt(seg['end'])}\n")
                f.write(f"{seg['text']}\n\n")
                
        #輸出TXT
        txt_path = os.path.join(audio_dir, f"{filename}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in segments_for_output:
                #每個片段換一行,若希望全部連在一起可移除 \n
                f.write(f"{seg['text']}\n")

        print(f"辨識完成,SRT 已輸出: {srt_path}")

        #--- 清理暫存檔與記憶體 ---
        #如果你想保留過濾後的音訊檔,請註解掉下面這幾行
        if os.path.exists(processed_file_path):
            try:
                #os.remove(processed_file_path) 
                pass
            except:
                pass

        del model
        del processor
        try:
            del vad_model
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        return [srt_path, Input_language]
    
    #ffmpeg升降調
    def change_pitch_ffmpeg(input_file:str, n_steps):
        if n_steps == 0:
            return False
        input_file = os.path.normpath(os.path.abspath(input_file))
        #計算頻率變化係數
        factor = 2 ** (n_steps / 12.0)
        dir_name = os.path.dirname(input_file)
        base_name = os.path.basename(input_file)
        name_no_ext, ext = os.path.splitext(base_name)
        
        if not ext: 
            ext = ".wav"
            
        output_file = os.path.join(dir_name, f"{name_no_ext}_step{n_steps}{ext}")
        # 組合濾鏡：
        # asetrate: 改變播放採樣率 (變調同時變速) -> 44100 * factor
        # atempo: 修正速度 (把速度變回去) -> 1 / factor
        # 注意：atempo 限制在 0.5 到 2.0 之間,若變調幅度過大需串聯多個 atempo,
        # 但一般升降調 (-12 ~ +12) 通常還在範圍內或只需簡單處理。
        
        # 為了簡化,假設採樣率為 44100 (或是讓 ffmpeg 自動處理,但 asetrate 需要數值)
        # 更穩健的方法是單純用 rubberband 濾鏡 (若 ffmpeg 有支援),
        # 這裡使用通用的 asetrate+atempo 方法：
        
        # 這裡稍微簡化,不指定絕對頻率,而是假設輸入是 44100 (標準 MP3/WAV)
        # 若要更精確,需先取得檔案的 sample rate,或是強制轉為 44100
        sample_rate = 44100
        new_rate = int(sample_rate * factor)
        tempo = 1.0 / factor
        
        filter_complex = f"aresample={sample_rate},asetrate={new_rate},atempo={tempo}"
        dir_name = os.path.dirname(input_file)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-af", filter_complex,
            output_file
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_file
        except subprocess.CalledProcessError as e:
            raise e