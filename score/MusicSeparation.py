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

#將 transformers 移到全域引用，解決 Worker Error
try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
except ImportError:
    print("錯誤：找不到 transformers 套件。請執行 pip install transformers accelerate")
    raise

#設定 Torchaudio 後端，解決 UserWarning (針對 Windows 環境)
if os.name == 'nt':
    try:
        torchaudio.set_audio_backend("soundfile")
    except:
        pass #新版 torchaudio 可能不支援此語法，則依賴預設

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
    def run_separation(audio_path, segment = 15.0, use_cuda = True) -> str|None:
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
    #沒少幻覺但原始方法
    """@staticmethod
    def run_ASR(audio="", Input_language=""):
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
        
        #0. 獲取音訊總長度 (用於防呆，避免時間軸超出)
        try:
            info = sf.info(file_path)
            audio_duration = info.duration
        except:
            audio_duration = 999999.0 #如果讀不到就設很大

        #--- VAD 處理 (程式碼保持不變) ---
        print("正在進行 VAD (語音活動偵測)...")
        processed_file_path = os.path.join(audio_dir, f"{filename}_vad_processed.wav")
        target_file_for_asr = file_path #預設為原檔

        try:
            #(VAD 載入與處理邏輯同前，省略以節省篇幅...)
            #這裡假設你的 VAD 程式碼正常運作，並生成了 processed_file_path
            #如果 VAD 成功:
            #target_file_for_asr = processed_file_path
            
            #--- 以下為簡化的 VAD 邏輯示意，請保留你原本完整的 VAD 區塊 ---
            vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
            wav = read_audio(file_path, sampling_rate=16000)
            speech_timestamps = get_speech_timestamps(wav, vad_model, threshold=0.35, sampling_rate=16000, min_speech_duration_ms=250, min_silence_duration_ms=100, speech_pad_ms=400)
            if len(speech_timestamps) > 0:
                vad_audio = torch.zeros_like(wav)
                for stamp in speech_timestamps:
                    vad_audio[stamp['start']:stamp['end']] = wav[stamp['start']:stamp['end']]
                save_audio(processed_file_path, vad_audio, sampling_rate=16000)
                target_file_for_asr = processed_file_path #★關鍵修正：確認使用處理過的檔案
                print(f"使用 VAD 處理後的檔案進行辨識: {target_file_for_asr}")
            else:
                print("VAD 未偵測到人聲，使用原檔。")
        except Exception as e:
            print(f"VAD 失敗，使用原檔: {e}")

        #--- 模型載入 ---
        tempmodel = "Zzzkay1/whisper-small-zh" 
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        try:
            processor = AutoProcessor.from_pretrained(tempmodel)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                tempmodel, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)
        except Exception as e:
            return None

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            torch_dtype=torch_dtype,
        )

        #★關鍵修正：參數調整以避免幻覺死循環
        generate_kwargs = {
            "task": "transcribe",
            "language": "zh",
            "repetition_penalty": 1.2,    #稍微降低一點，太高有時會導致跳字
            "no_speech_threshold": 0.3,   #提高對靜音的敏感度
            "condition_on_prev_tokens": False, #★★★ 最重要：設為 False 防止依賴前句產生的死循環
            "compression_ratio_threshold": 1.35, #如果壓縮率太高(代表文字一直重複)，則捨棄該段
            "temperature": 0.2, #給一點點隨機性，避免陷入決定性的死路
            "logprob_threshold": -1.0, 
        }

        print("開始辨識...")
        try:
            #★關鍵修正：確保這裡傳入的是 target_file_for_asr (VAD後的檔案)
            results_dict = pipe(
                target_file_for_asr, 
                return_timestamps=True,
                generate_kwargs=generate_kwargs
            )
        except Exception as e:
            print(f"辨識錯誤: {e}")
            return None

        #--- 資料整理與後處理 (過濾重複 + 時間檢查) ---
        chunks = results_dict.get("chunks", [])
        segments_for_output = []
        last_text = ""  #紀錄上一句文字

        for chunk in chunks:
            if 'timestamp' in chunk and 'text' in chunk:
                start, end = chunk['timestamp']
                text = chunk['text'].strip()

                #1. 修正時間戳為 None 的情況
                if start is None: continue
                if end is None: end = start + 2.0

                #2. ★硬性檢查：如果開始時間超過音訊總長，直接結束迴圈
                if start > audio_duration:
                    break
                #如果結束時間超過總長，修正為總長
                if end > audio_duration:
                    end = audio_duration

                #3. ★過濾邏輯：移除完全重複的句子
                if text == last_text:
                    continue
                
                #4. 過濾空字串或極短的無意義字元
                if len(text) == 0:
                    continue

                segments_for_output.append({
                    "start": start,
                    "end": end,
                    "text": text
                })
                last_text = text #更新上一句

        #--- (後續的 SRT/VTT/TXT 輸出程式碼保持不變) ---
        #... 這裡放原來的 format_time_srt 等輸出邏輯 ...
        
        #這裡為了完整性補上輸出部分
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

        #... (清理記憶體代碼保持不變) ...
        del model
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        return [srt_path, "zh"]"""
    #少幻覺但一段超長
    """@staticmethod
    def run_ASR(audio="", Input_language=""):
        import torch
        import torchaudio
        import gc
        import os
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        
        #... (GUI 與 檔案選擇程式碼保持不變) ...
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
        
        #--- 1. 載入 Whisper 模型 (回歸最單純的載入方式) ---
        tempmodel = "Zzzkay1/whisper-small-zh" 
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

        #--- 2. 使用 Silero VAD 獲取精確的時間戳記 ---
        print("正在進行 VAD 分析並切割音訊...")
        try:
            vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
            
            #讀取完整音訊
            wav = read_audio(file_path, sampling_rate=16000)
            
            #★ 參數調整：確保不要切太碎，也不要太長
            speech_timestamps = get_speech_timestamps(
                wav, 
                vad_model, 
                threshold=0.4,          #適中的閾值
                sampling_rate=16000, 
                min_speech_duration_ms=250, 
                min_silence_duration_ms=300, #稍微縮短，讓斷句更頻繁
                speech_pad_ms=100       #減少 padding，避免吃到前後雜訊
            )
        except Exception as e:
            print(f"VAD 失敗: {e}")
            return None

        #--- 3. 手動切割並逐段辨識 (解決死循環的關鍵) ---
        segments_for_output = []
        print(f"共偵測到 {len(speech_timestamps)} 個語音片段，開始逐段辨識...")

        #獲取 input_features 的輔助函式
        def transcribe_segment(segment_waveform):
            input_features = processor(
                segment_waveform, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(device).to(torch_dtype)
            
            #★ 參數回歸穩定設定 (不要設太極端)
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
            
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=250,      #限制單句最大長度，防止無限輸出
                    no_repeat_ngram_size=3,  #輕微防止重複 (比 repetition_penalty 溫和)
                    temperature=0.2,         #給一點點彈性，避免死路
                    do_sample=True,          #允許採樣
                    top_k=50,                #限制採樣範圍
                )
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription.strip()

        #遍歷每一個 VAD 切出來的時間段
        for idx, stamp in enumerate(speech_timestamps):
            start_sample = stamp['start']
            end_sample = stamp['end']
            
            #轉換為秒數
            start_time = start_sample / 16000
            end_time = end_sample / 16000
            
            #提取這一段的音訊 (Tensor)
            segment_wav = wav[start_sample:end_sample]
            
            #如果片段太短(<0.2s)，直接跳過
            if (end_time - start_time) < 0.2:
                continue

            #進行辨識
            #注意：這裡輸入的是 numpy array (轉成 tensor 在 transcribe_segment 內處理)
            try:
                text = transcribe_segment(segment_wav.numpy())
            except:
                continue #如果轉換失敗就跳過
            
            #--- 後處理過濾 ---
            #1. 過濾重複詞 (針對 "一路逛一路逛" 這種句子內部重複)
            if len(text) > 10 and len(set(text)) < 4: #如果字很長但用到的字元種類很少
                continue
                
            #2. 過濾極短無意義文字
            if len(text) < 2:
                continue
            
            #3. 簡單的去重 (跟上一句比)
            if segments_for_output and text == segments_for_output[-1]['text']:
                continue

            print(f"[{start_time:.2f}s -> {end_time:.2f}s] {text}")
            
            segments_for_output.append({
                "start": start_time,
                "end": end_time,
                "text": text
            })

        #--- 輸出邏輯 (SRT/VTT) ---
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

        print(f"辨識完成，已輸出 SRT: {srt_path}")

        #--- 清理 ---
        del model
        del processor
        try:
            del vad_model
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        return [srt_path, "zh"]"""
    #不能跑，超爛
    """@staticmethod
    def run_ASR(audio="", Input_language=""):
        import torch
        import gc
        import os
        import soundfile as sf
        import math
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
        
        #... (GUI 與 檔案選擇程式碼保持不變) ...
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
        
        #--- 1. 載入模型 ---
        tempmodel = "Zzzkay1/whisper-small-zh" 
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"正在載入模型 {tempmodel}...")
        try:
            processor = AutoProcessor.from_pretrained(tempmodel)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                tempmodel, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)
            
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
                torch_dtype=torch_dtype,
            )
        except Exception as e:
            print(f"模型載入失敗: {e}")
            return None

        #--- 2. VAD 切割 ---
        print("正在進行 VAD 分析...")
        try:
            vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
            
            wav = read_audio(file_path, sampling_rate=16000)
            
            #這裡我們放寬一點，讓 VAD 盡量抓，後面我們會手動強制切
            speech_timestamps = get_speech_timestamps(
                wav, 
                vad_model, 
                threshold=0.4,           
                sampling_rate=16000, 
                min_speech_duration_ms=250, 
                min_silence_duration_ms=100, 
                speech_pad_ms=50       
            )
        except Exception as e:
            print(f"VAD 失敗: {e}")
            return None

        #--- 3. 處理與辨識 (Micro-Chunking) ---
        segments_for_output = []
        print(f"VAD 初步切出 {len(speech_timestamps)} 個片段，正在進行長度檢查與辨識...")

        temp_segment_file = os.path.join(audio_dir, "temp_segment_processing.wav")

        generate_kwargs = {
            "task": "transcribe",
            "language": "zh",
            "condition_on_prev_tokens": False, #避免死循環
            "repetition_penalty": 1.2,
            "no_speech_threshold": 0.4,
            "temperature": 0.0
        }

        #★★★ 關鍵邏輯：強制切分過長的片段 ★★★
        MAX_CHUNK_DURATION = 25.0  #設定上限為 25 秒 (留一點緩衝給 30 秒限制)

        final_chunks_to_process = []

        #第一階段：預處理時間戳，把過長的切短
        for stamp in speech_timestamps:
            start_sample = stamp['start']
            end_sample = stamp['end']
            duration_sec = (end_sample - start_sample) / 16000

            if duration_sec <= MAX_CHUNK_DURATION:
                #如果長度 OK，直接加入
                final_chunks_to_process.append((start_sample, end_sample))
            else:
                #如果太長，強制切分成多段
                #print(f"發現超長片段 ({duration_sec:.2f}s)，進行強制切分...")
                num_sub_chunks = math.ceil(duration_sec / MAX_CHUNK_DURATION)
                chunk_samples = int(MAX_CHUNK_DURATION * 16000)
                
                for i in range(num_sub_chunks):
                    sub_start = start_sample + (i * chunk_samples)
                    sub_end = min(sub_start + chunk_samples, end_sample)
                    
                    #避免切出極短的尾巴
                    if (sub_end - sub_start) / 16000 < 0.2:
                        continue
                        
                    final_chunks_to_process.append((sub_start, sub_end))

        #第二階段：開始辨識
        print(f"長度檢查完成，共需處理 {len(final_chunks_to_process)} 個微片段。")

        for idx, (start_sample, end_sample) in enumerate(final_chunks_to_process):
            
            vad_start_time = start_sample / 16000
            vad_end_time = end_sample / 16000
            
            segment_wav = wav[start_sample:end_sample].numpy()
            
            #存成暫存檔
            try:
                sf.write(temp_segment_file, segment_wav, 16000)
            except Exception as e:
                continue

            #辨識
            try:
                #★ return_timestamps=False 避開 logprobs bug
                #★ 因為我們強制切到 25s 以下，所以不會觸發 >30s bug
                result = pipe(
                    temp_segment_file,
                    return_timestamps=False, 
                    generate_kwargs=generate_kwargs
                )
                text = result['text'].strip()
            except Exception as e:
                #如果真的還是失敗，就印出來但不要中斷整個程式
                print(f"片段辨識略過 ({vad_start_time:.1f}s): {e}")
                continue

            if len(text) < 1: continue
            if segments_for_output and text == segments_for_output[-1]['text']: continue

            segments_for_output.append({
                "start": vad_start_time,
                "end": vad_end_time,
                "text": text
            })

            #顯示進度
            if idx % 5 == 0:
                print(f"進度 {idx}/{len(final_chunks_to_process)}: {text[:10]}...")

        #--- 清理暫存檔 ---
        if os.path.exists(temp_segment_file):
            try:
                os.remove(temp_segment_file)
            except:
                pass

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

        print(f"辨識完成，SRT 已輸出: {srt_path}")

        #--- 清理記憶體 ---
        del model
        del pipe
        try:
            del vad_model
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        return [srt_path, "zh"]"""
    #沒過濾但工作正常
    """
    @staticmethod
    def run_ASR(audio="", Input_language=""):
        import torch
        import gc
        import os
        import math
        import soundfile as sf
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        
        #... (GUI 與 檔案選擇程式碼保持不變) ...
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
        
        #--- 1. 載入模型 (不使用 Pipeline) ---
        tempmodel = "Zzzkay1/whisper-small-zh" 
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"正在載入模型 {tempmodel} (Direct Mode)...")
        try:
            processor = AutoProcessor.from_pretrained(tempmodel)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                tempmodel, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)
            #注意：這裡不再建立 pipe = pipeline(...)
        except Exception as e:
            print(f"模型載入失敗: {e}")
            return None

        #--- 2. VAD 切割 ---
        print("正在進行 VAD 分析...")
        try:
            vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
            
            wav = read_audio(file_path, sampling_rate=16000)
            
            speech_timestamps = get_speech_timestamps(
                wav, 
                vad_model, 
                threshold=0.4,           
                sampling_rate=16000, 
                min_speech_duration_ms=250, 
                min_silence_duration_ms=100, 
                speech_pad_ms=50       
            )
        except Exception as e:
            print(f"VAD 失敗: {e}")
            return None

        #--- 3. 處理與辨識 (Direct Generate) ---
        segments_for_output = []
        print(f"VAD 初步切出 {len(speech_timestamps)} 個片段，正在進行長度檢查與辨識...")

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
                num_sub_chunks = math.ceil(duration_sec / MAX_CHUNK_DURATION)
                chunk_samples = int(MAX_CHUNK_DURATION * 16000)
                for i in range(num_sub_chunks):
                    sub_start = start_sample + (i * chunk_samples)
                    sub_end = min(sub_start + chunk_samples, end_sample)
                    if (sub_end - sub_start) / 16000 < 0.2: continue
                    final_chunks_to_process.append((sub_start, sub_end))

        print(f"準備處理 {len(final_chunks_to_process)} 個微片段。")

        #設定解碼參數 (固定中文)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

        for idx, (start_sample, end_sample) in enumerate(final_chunks_to_process):
            
            vad_start_time = start_sample / 16000
            vad_end_time = end_sample / 16000
            
            #提取音訊 (Tensor -> Numpy)
            segment_wav = wav[start_sample:end_sample].numpy()
            
            #★★★ 關鍵修改：直接使用 model.generate ★★★
            try:
                #1. 將音訊轉為輸入特徵 (Input Features)
                input_features = processor(
                    segment_wav, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(device).to(torch_dtype)

                #2. 直接生成 Token IDs
                #這裡完全繞過了 Pipeline 的 logprobs 變數檢查，所以絕對不會崩潰
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_features,
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255, #限制長度
                        temperature=0.0,    #降低幻覺
                        repetition_penalty=1.2
                    )

                #3. 解碼回文字
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            except Exception as e:
                print(f"片段辨識例外錯誤 ({vad_start_time:.1f}s): {e}")
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

        print(f"辨識完成，SRT 已輸出: {srt_path}")

        #--- 清理記憶體 ---
        del model
        del processor
        try:
            del vad_model
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        return [srt_path, "zh"]
        """
    @staticmethod
    def run_ASR(audio="", Input_language=""):
        import torch
        import gc
        import os
        import math
        import soundfile as sf
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        
        #... (GUI 與 檔案選擇程式碼保持不變) ...
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
        
        #0. 獲取音訊總長度
        try:
            info = sf.info(file_path)
            audio_duration = info.duration
        except:
            audio_duration = 999999.0

        #--- 1. 載入模型 (Direct Mode) ---
        tempmodel = "Zzzkay1/whisper-small-zh" 
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

        #--- 2. VAD 預處理與過濾 (加強版：殺死孤島雜訊) ---
        print("正在進行 VAD 過濾 (將非人聲部分靜音)...")
        processed_file_path = os.path.join(audio_dir, f"{filename}_vad_processed.wav")
        
        try:
            vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
            
            #讀取原始音訊
            wav = read_audio(file_path, sampling_rate=16000)
            
            #取得原始的人聲時間戳
            raw_timestamps = get_speech_timestamps(
                wav, 
                vad_model, 
                threshold=0.4,           
                sampling_rate=16000, 
                min_speech_duration_ms=100, #先設小一點，讓我們能捕捉到所有東西，後面再過濾
                min_silence_duration_ms=100, 
                speech_pad_ms=50       
            )

            #★★★ 新增：清洗時間戳記邏輯 ★★★
            clean_timestamps = []
            
            for stamp in raw_timestamps:
                start = stamp['start']
                end = stamp['end']
                duration_ms = (end - start) / 16000 * 1000
                
                #策略 1: 殺死孤島
                #如果這段聲音小於 400ms (0.4秒)，人類講話很少這麼短 (通常是雜音)現在改500ms，出事再改回去
                if duration_ms < 500: 
                    print(f"發現短雜訊，已過濾: {duration_ms:.2f}ms")
                    continue
                
                clean_timestamps.append(stamp)

            #使用清洗後的時間戳
            speech_timestamps = clean_timestamps

            if len(speech_timestamps) > 0:
                vad_audio = torch.zeros_like(wav)
                
                #只複製清洗後的片段
                for stamp in speech_timestamps:
                    vad_audio[stamp['start']:stamp['end']] = wav[stamp['start']:stamp['end']]
                
                #儲存 (建議保留這行查看處理後的檔案是否還有那個小點)
                save_audio(processed_file_path, vad_audio, sampling_rate=16000)
                print(f"VAD 過濾完成，暫存檔: {processed_file_path}")
                
                wav = vad_audio 
            else:
                print("警告: VAD 過濾後沒有剩餘人聲 (可能都是雜訊)，將使用原始檔案。")

        except Exception as e:
            print(f"VAD 處理失敗，將使用原始檔案: {e}")
            #如果失敗，wav 保持原樣，繼續執行

        #--- 3. 微切分與辨識 (Direct Generate) ---
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
                #如果單一句子超過 25 秒，強制切斷
                num_sub_chunks = math.ceil(duration_sec / MAX_CHUNK_DURATION)
                chunk_samples = int(MAX_CHUNK_DURATION * 16000)
                for i in range(num_sub_chunks):
                    sub_start = start_sample + (i * chunk_samples)
                    sub_end = min(sub_start + chunk_samples, end_sample)
                    if (sub_end - sub_start) / 16000 < 0.2: continue
                    final_chunks_to_process.append((sub_start, sub_end))

        #設定解碼參數
        #使用 Input_language 參數，若為 None 或 "auto" 則自動偵測
        lang = Input_language if Input_language and Input_language.lower() != "auto" else None
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")

        for idx, (start_sample, end_sample) in enumerate(final_chunks_to_process):
            
            vad_start_time = start_sample / 16000
            vad_end_time = end_sample / 16000
            
            #★ 這裡取出的 segment_wav 已經是經過 VAD 過濾的乾淨音訊了
            segment_wav = wav[start_sample:end_sample].numpy()
            
            try:
                #1. 轉特徵
                input_features = processor(
                    segment_wav, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(device).to(torch_dtype)

                #2. 直接生成 (Direct Generate)
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_features,
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255, 
                        temperature=0.0,
                        repetition_penalty=1.2
                    )

                #3. 解碼
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
                #每個片段換一行，若希望全部連在一起可移除 \n
                f.write(f"{seg['text']}\n")

        print(f"辨識完成，SRT 已輸出: {srt_path}")

        #--- 清理暫存檔與記憶體 ---
        #如果你想保留過濾後的音訊檔，請註解掉下面這幾行
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

        return [srt_path, "zh"]