import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import builtins

# 防止 Windows 下終端機預設編碼 (cp950) 無法印出日文或特殊符號而導致崩潰
_orig_print = builtins.print
def safe_print(*args, **kwargs):
    try:
        _orig_print(*args, **kwargs)
    except UnicodeEncodeError:
        new_args = []
        enc = sys.stdout.encoding if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding else 'cp950'
        for arg in args:
            if isinstance(arg, str):
                new_args.append(arg.encode(enc, 'replace').decode(enc, 'replace'))
            else:
                new_args.append(arg)
        _orig_print(*new_args, **kwargs)
builtins.print = safe_print

import base64
import glob
import shutil
import time
import threading
import re
import json
from datetime import datetime
from google import genai
from google.genai import types
from YouTubeDownload import YouTubeDownload
from MusicTools import MusicTools

# 啟動本地 HTTP Server 避開 Streamlit 靜態檔案 1GB 限制
import tornado.ioloop
import tornado.web
import asyncio
import socket

def get_local_ip():
    """取得本機在區域網路中的 IP 位址，讓區網其他裝置能正確連線"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def get_free_port(start_port=8000):
    for port in range(start_port, 8100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

def start_local_file_server(directory, port=8000):
    """啟動一個背景 Tornado HTTP Server 專門提供靜態檔案 (完美支援 Range Requests)"""
    free_port = get_free_port(port)
    if not free_port:
        return None

    class CORSStaticFileHandler(tornado.web.StaticFileHandler):
        def set_default_headers(self):
            self.set_header("Access-Control-Allow-Origin", "*")
            self.set_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.set_header("Access-Control-Allow-Headers", "Range")
            self.set_header("Access-Control-Expose-Headers", "Accept-Ranges, Content-Encoding, Content-Length, Content-Range")
            self.set_header("Cache-Control", "no-store, no-cache, must-revalidate")
            
        def options(self, *args, **kwargs):
            self.set_status(204)
            self.finish()

    app = tornado.web.Application([
        (r"/(.*)", CORSStaticFileHandler, {"path": directory})
    ])
    
    def run_server():
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            app.listen(free_port)
            tornado.ioloop.IOLoop.current().start()
        except Exception as e:
            print(f"Tornado Server Error: {e}")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return free_port

@st.cache_resource
def get_local_server_port():
    return start_local_file_server("static/output")

#---Gemini 字幕校正工具函式---
def parse_srt_for_correction(file_path):
    """讀取並解析 SRT 檔案，抽離時間軸與文字"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'\n\n+', content.strip())
    subtitles = []
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            index = lines[0].strip()
            timestamp = lines[1].strip()
            text = " ".join(lines[2:]).strip() 
            subtitles.append({
                "index": index,
                "timestamp": timestamp,
                "text": text
            })
            
    return subtitles

def write_srt_for_correction(subtitles, output_path):
    """將修正後的字幕陣列重新組裝成 SRT 檔案"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sub in subtitles:
            f.write(f"{sub['index']}\n")
            f.write(f"{sub['timestamp']}\n")
            f.write(f"{sub['text']}\n\n")

def correct_srt_text_only(input_file: str, output_file: str, api_key: str = None):
    # 嘗試從 API_Key.txt 讀取
    if not api_key and os.path.exists("API_Key.txt"):
        with open("API_Key.txt", "r", encoding="utf-8") as f:
            api_key = f.read().strip()

    # 初始化 Gemini 客戶端
    if api_key:
        client = genai.Client(api_key=api_key)
    elif os.environ.get("GEMINI_API_KEY"):
        client = genai.Client()
    else:
        return False, "未設定 Gemini API Key，請在左側邊欄輸入或建立 API_Key.txt。"
        
    subtitles = parse_srt_for_correction(input_file)
    
    if not subtitles:
        return False, "解析 SRT 失敗或檔案為空。"

    # 製作專屬給 LLM 閱讀的 Payload (只包含 行號 與 文字)
    llm_payload = ""
    for sub in subtitles:
        if sub['text']: 
            llm_payload += f"{sub['index']}|||{sub['text']}\n"

    system_instruction = """
    You are an expert lyric proofreader. Your task is to correct homophone errors, mondegreens (misheard lyrics), and typos in the provided lyrics based on context.

    ### STRICT FORMATTING RULES:
    1. Each line of your input and output must strictly follow the format: `LineNumber|||LyricText`
    2. You MUST NOT merge, delete, or add any lines. The number of output lines must EXACTLY match the input lines.
    3. Keep the `LineNumber` and `|||` separator exactly as provided.
    4. DO NOT output any markdown tags (like ```), greetings, or explanations. ONLY output the corrected lines.
    5. If a line requires no correction, output the original line exactly as it was.

    ### LINGUISTIC RULES:
    1. Correct obvious homophone typos or misheard phrases based on context (e.g., fixing "Bed for long the cross" to a contextually correct lyric).
    2. STRICTLY respect the original language and dialects (e.g., Traditional Chinese, Taiwanese Hokkien, English). 
    3. DO NOT translate Taiwanese Hokkien into standard Mandarin. ONLY fix phonetic/homophone typos while preserving the Hokkien grammar and vocabulary.

    ### EXAMPLE:
    【Input】
    1|||因位我剛好欲見你
    2|||這是一首簡單的小情歌
    3|||Bed for long the cross

    【Output】
    1|||因為我剛好遇見你
    2|||這是一首簡單的小情歌
    3|||Battle on the cross
    """

    try:
        max_retries = 3
        retry_delay = 5
        response = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=[system_instruction, f"【原始歌詞資料】：\n{llm_payload}"],
                    config=types.GenerateContentConfig(
                        temperature=0.1, 
                    )
                )
                break
            except Exception as api_err:
                # 處理伺服器忙碌(503)、請求過多(429)或內部錯誤(500)的情況
                if ("503" in str(api_err) or "429" in str(api_err) or "500" in str(api_err) or "UNAVAILABLE" in str(api_err)) and attempt < max_retries - 1:
                    import time
                    print(f"API 忙碌或伺服器錯誤 ({api_err})，等待 {retry_delay} 秒後重試... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise api_err
        
        corrected_text = response.text.strip()
        
        # 移除可能殘留的 markdown 標記
        if corrected_text.startswith("```"):
            corrected_text = "\n".join(corrected_text.split("\n")[1:-1]).strip()

        # 解析 LLM 回傳的結果，建立 {行號: 修正後歌詞} 的字典
        corrected_dict = {}
        for line in corrected_text.split('\n'):
            if '|||' in line:
                parts = line.split('|||', 1) 
                if len(parts) == 2:
                    idx = parts[0].strip()
                    text = parts[1].strip()
                    corrected_dict[idx] = text

        # 將修正後的歌詞「重新注入」回原本的 SRT 結構中
        modify_count = 0
        for sub in subtitles:
            idx = sub['index']
            if idx in corrected_dict:
                original_text = sub['text']
                new_text = corrected_dict[idx]
                
                if original_text != new_text:
                    modify_count += 1
                    
                sub['text'] = new_text

        # 寫出最終的 SRT 檔案
        write_srt_for_correction(subtitles, output_file)
        
        # 同時將純歌詞寫出為 txt 檔案
        txt_output_path = os.path.splitext(output_file)[0] + "_fix" + ".txt"
        with open(txt_output_path, 'w', encoding='utf-8') as txt_f:
            for sub in subtitles:
                if sub['text'].strip(): # 略過空白行
                    txt_f.write(f"{sub['text']}\n")

        return True, f"共修正了 {modify_count} 句歌詞。純文字歌詞已儲存。"

    except Exception as e:
        return False, f"發生錯誤: {e}"

#---核心邏輯與工具函式---

def parse_srt(srt_path):
    """讀取 SRT 檔案並解析為 JSON 格式供 JS 使用"""
    if not os.path.exists(srt_path):
        return []
    
    content = ""
    encodings = ['utf-8', 'cp950', 'gbk', 'utf-16']
    for enc in encodings:
        try:
            with open(srt_path, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    
    if not content:
        return []

    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:(?!\n\n).)*)', re.DOTALL)
    matches = pattern.findall(content)
    
    subtitles = []
    for match in matches:
        index, start_str, end_str, text = match
        
        def time_to_seconds(t_str):
            h, m, s = t_str.split(':')
            s, ms = s.split(',')
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
            
        start = time_to_seconds(start_str)
        end = time_to_seconds(end_str)
        clean_text = text.strip().replace('\n', '<br>')
        
        subtitles.append({
            "start": start,
            "end": end,
            "text": clean_text
        })
    return subtitles
#轉碼影片
def get_video_base64(video_path):
    try:
        with open(video_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""

#取得輸出資料夾內檔案
def get_processed_files(output_dir):
    if not os.path.exists(output_dir):
        return []
    files = glob.glob(os.path.join(output_dir, "*.mp4"))
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.basename(f) for f in files]
#清除輸出資料夾
def clear_output_folder(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)

#---背景任務管理---
@st.cache_resource
class TaskManager:
    def __init__(self):
        self.is_processing = False
        self.current_task_url = None
        self.last_completed_file = None
        self.status_message = ""
        self.lock = threading.Lock()

    #輸入網址開始執行
    def start_task(self, url, output_dir, pitch_steps=0, input_language="zh", use_gemini=True, api_key=None):
        with self.lock:
            if self.is_processing:
                return False
            
            self.is_processing = True
            self.current_task_url = url
            self.status_message = "啟動處理程序..."
            
            thread = threading.Thread(target=self._worker, args=(url, output_dir, pitch_steps, input_language, use_gemini, api_key))
            thread.daemon = True
            thread.start()
            return True
    
    #開始執行的具體步驟
    def _worker(self, url, output_dir, pitch_steps, input_language:str, use_gemini:bool, api_key:str):
        if (not(YouTubeDownload.check_YouTube_available(url))):
            self.status_message = "連結輸入錯誤或網路不可用"
            return
        try:
            self.status_message = "下載 YouTube 音訊與視訊..."
            audio = YouTubeDownload.download_youtube_audio(url)
            video = YouTubeDownload.download_youtube_video(url)
            
            self.status_message = "進行人聲分離..."
            # 透過獨立行程執行，避免鎖死 GIL
            separation = self._run_worker_subprocess("separation", audio)
            if not separation:
                self.status_message = "✗ 處理失敗 (人聲分離錯誤)"
                return

            target_audio = separation[0] 
            
            #變調處理
            if pitch_steps != 0:
                self.status_message = f"進行升降調處理({pitch_steps}半音)..."
                
                #定義變調後的檔名
                path_parts = os.path.splitext(target_audio)
                shifted_audio_path = f"{path_parts[0]}_pitch{pitch_steps}{path_parts[1]}"
                
                #FFmpeg變調
                target_audio = MusicTools.change_pitch_ffmpeg(target_audio, pitch_steps)

            self.status_message = "合併影片與音訊..."

            #使用變調後的音訊進行合併
            merge = YouTubeDownload.merge_video_audio(video, target_audio)

            self.status_message = f"Whisper 歌詞辨識中(語言: {input_language})..."
            # 透過獨立行程執行，避免鎖死 GIL
            AsrReturn = self._run_worker_subprocess("asr", separation[1], input_language)
            
            srt_source_path = None
            if AsrReturn and len(AsrReturn) > 0:
                srt_source_path = AsrReturn[0]

            merge = merge.strip('"').strip("'")
            safe_name = os.path.basename(merge).replace("／", "_").replace("/", "_").replace("\\", "_")
            safe_name = re.sub(r'[\\/*?:"<>|]', "", safe_name)

            final_name_base = os.path.splitext(safe_name)[0]
            file_extension = os.path.splitext(safe_name)[1] if os.path.splitext(safe_name)[1] else ".mp4"

            #加上升降調數字
            final_name_base = f"{final_name_base}_{pitch_steps}"
            safe_name = f"{final_name_base}{file_extension}"

            #檢查檔名是否存在，存在加時間碼
            if os.path.exists(os.path.join(output_dir, safe_name)):
                timestamp = datetime.now().strftime("%H%M%S")
                final_name_base = f"{final_name_base}_{timestamp}"
                safe_name = f"{final_name_base}{file_extension}"
            
            #最終路徑
            out_video_path = os.path.join(output_dir, safe_name)
            out_srt_path = os.path.join(output_dir, f"{final_name_base}.srt")
            
            # 確保資料夾存在以防 WinError 3
            os.makedirs(output_dir, exist_ok=True)
            
            if os.path.exists(merge):
                if os.path.abspath(merge) != os.path.abspath(out_video_path):
                    if os.path.exists(out_video_path): os.remove(out_video_path)
                    shutil.move(merge, out_video_path)
            else:
                with open(out_video_path, 'w') as f: f.write("dummy content")

            if srt_source_path and os.path.exists(srt_source_path):
                if os.path.exists(out_srt_path): os.remove(out_srt_path)
                shutil.move(srt_source_path, out_srt_path)
                
                # 在這裡進行歌詞 AI 校正
                correction_msg = ""
                gemini_success = True
                if use_gemini:
                    self.status_message = "進行 AI 歌詞校正中..."
                    gemini_success, msg = correct_srt_text_only(out_srt_path, out_srt_path, api_key)
                    print(f"correct_srt_text_only 執行結果 -> success: {gemini_success}, msg: {msg}")
                    correction_msg = f"\nAI校正：{msg}"
                    if not gemini_success:
                        print(f"Gemini correction failed: {msg}")

            self.last_completed_file = safe_name
            if use_gemini and not gemini_success:
                self.status_message = f"✗ 影片已處理，但AI校正失敗：{safe_name}{correction_msg}"
            else:
                self.status_message = f"✓ 完成：{safe_name}{correction_msg}"
            
        except Exception as e:
            self.status_message = f"✗ 錯誤: {str(e)}"
            print(f"Worker Error: {e}")
        finally:
            with self.lock:
                self.is_processing = False

    def _run_worker_subprocess(self, task, *args):
        import subprocess
        import sys
        import json
        import os
        
        worker_script = os.path.join(os.path.dirname(__file__), "worker.py")
        cmd = [sys.executable, worker_script, task] + list(args)
        
        try:
            # 確保能正確解析 JSON 與中文，加入 errors='replace' 防止非 UTF-8 輸出導致 _readerthread 崩潰
            # 同時設定 PYTHONIOENCODING 確保 worker.py 輸出純 UTF-8
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                encoding='utf-8', 
                errors='replace', 
                env=env,
                bufsize=1
            )
            
            out_text = ""
            for line in process.stdout:
                print(line, end="", flush=True)
                out_text += line
                
            process.wait()
            
            if process.returncode != 0:
                print(f"Subprocess failed with return code {process.returncode}")
                return None
            
            if out_text and "---RESULT_START---" in out_text and "---RESULT_END---" in out_text:
                json_str = out_text.split("---RESULT_START---")[1].split("---RESULT_END---")[0].strip()
                return json.loads(json_str)
            else:
                print(f"Worker Output/Error:\n{out_text}")
                return None
                
        except Exception as e:
            print(f"Subprocess failed with error: {e}")
            return None

task_manager = TaskManager()

@st.cache_data(show_spinner=False, ttl=3600)
def get_cached_recommendations(song_name, count=3):
    return YouTubeDownload.get_recommendations_by_artist(song_name, count=count)

#---介面顯示類別---

class WebDisplay:
    #初始化
    def __init__(self):
        self.output_dir = "static/output"
        os.makedirs(self.output_dir, exist_ok=True)

        if 'is_initialized' not in st.session_state:
            st.session_state.is_initialized = True
            st.session_state.current_playing_name = None

    #左方佇列
    def sidebar_queue_fragment(self):
        if task_manager.is_processing:
            self._sidebar_queue_polling()
        else:
            self._sidebar_queue_static()

    @st.fragment(run_every=2)
    def _sidebar_queue_polling(self):
        self._render_queue_content()
        # 當任務完成時，觸發整頁重整以跳出自動輪詢的迴圈
        if not task_manager.is_processing:
            st.rerun()

    @st.fragment
    def _sidebar_queue_static(self):
        self._render_queue_content()

    def _render_queue_content(self):
        st.header("處理列表")
        
        #是否執行中
        if task_manager.is_processing:
            st.info(f"{task_manager.status_message}")
            st.spinner("正在處理中...") 
        elif task_manager.status_message and "✓" in task_manager.status_message:
            st.success(task_manager.status_message)
        elif task_manager.status_message and "✗" in task_manager.status_message:
            st.error(task_manager.status_message)
        
        file_list = get_processed_files(self.output_dir)
        
        if file_list:
            for filename in file_list:
                is_selected = (st.session_state.current_playing_name == filename)
                label = f"{'▶ ' if is_selected else ''}{filename}"
                if filename == task_manager.last_completed_file:
                    label += " (New!)"

                if st.button(label, key=f"btn_{filename}", width='stretch'):
                    st.session_state.current_playing_name = filename
                    st.rerun()
        else:
            st.caption("尚無檔案，請新增任務。")
        
        st.markdown("---")
        if st.button("清除所有紀錄", width='stretch'):
            clear_output_folder(self.output_dir)
            st.session_state.current_playing_name = None
            task_manager.status_message = ""
            task_manager.last_completed_file = None
            st.rerun()

    #主畫面
    def render(self):
        #使用 wide layout 讓左右分欄更寬敞
        st.set_page_config(layout="wide", page_title="AI 影片處理與字幕")
        
        st.markdown("""
            <style>
            .stButton button { text-align: left; }
            div[data-testid="stStatusWidget"] { visibility: hidden; }
            /* 調整頂部間距，因為拿掉了標題 */
            .block-container { padding-top: 2rem; } 
            </style>
        """, unsafe_allow_html=True)

        #側邊欄
        with st.sidebar:
            default_api_key = ""
            if os.path.exists("API_Key.txt"):
                with open("API_Key.txt", "r", encoding="utf-8") as f:
                    default_api_key = f.read().strip()
            
            st.text_input("Gemini API Key", value=default_api_key, type="password", key="gemini_api_key", help="若要使用 AI 校正，請輸入 API Key (也可以設定於環境變數 GEMINI_API_KEY 或建立 API_Key.txt)")
            sidebar_placeholder = st.empty()

        #輸入框
        #建立一個容器放在最上方
        with st.container():
            with st.form("task_form"):
                c1, c4, c2, c_gemini, c3 = st.columns([4, 1.5, 1.5, 1.5, 2], vertical_alignment="bottom") #調整欄位比例
                with c1:
                    url_input = st.text_input("YouTube URL 或 關鍵字", placeholder="請輸入 YouTube 連結或歌曲關鍵字...", label_visibility="collapsed", key="url_input_key")
                with c4:
                    #建立語言映射字典
                    lang_mapping = {"中文": "zh", "英文": "en", "閩南語": "nan"}
                    selected_lang = st.selectbox("辨識語言", options=list(lang_mapping.keys()))
                with c2:
                    #升降調選擇(-6到+6半音)
                    semitones = st.number_input("升降調 (半音)", min_value=-12, max_value=12, value=0, step=1, help="正數為升調，負數為降調")
                with c_gemini:
                    use_gemini = st.checkbox("上下文歌詞校正", value=True, key="use_gemini_chk")
                with c3:
                    submitted = st.form_submit_button("加入背景處理", type="primary", width='stretch')
                
                if submitted:
                    if url_input:
                        if not task_manager.is_processing:
                            with st.spinner("🔍 正在搜尋影片..."):
                                actual_url = YouTubeDownload.get_actual_url(url_input)
                                
                            if actual_url:
                                current_api_key = st.session_state.get("gemini_api_key", "").strip()
                                success = task_manager.start_task(actual_url, self.output_dir, semitones, selected_lang, use_gemini, current_api_key)
                                if success:
                                    st.toast(f"已加入排程(變調:{semitones})，請留意側邊欄狀態！")
                                else:
                                    st.warning("任務啟動失敗。")
                            else:
                                st.warning("找不到相關影片，請更換關鍵字。")
                        else:
                            st.warning("目前已有任務正在執行，請稍候。")
                    else:
                        st.warning("請輸入有效的 URL 或關鍵字")

        st.markdown("---") #分隔線

        #分欄:左播放器，右歌詞
        #調整比例
        col_player, col_lyrics = st.columns([1.6, 1]) 

        target_file = st.session_state.current_playing_name
        
        #預先處理好路徑和資料
        video_url = ""
        subtitles_json = "[]"
        has_video = False

        if target_file:
            video_path = os.path.join(self.output_dir, target_file)
            srt_path = os.path.splitext(video_path)[0] + ".srt"
            
            if os.path.exists(video_path):
                local_port = get_local_server_port()
                import urllib.parse
                if local_port:
                    host_ip = get_local_ip()
                    video_url = f"http://{host_ip}:{local_port}/{urllib.parse.quote(target_file)}"
                else:
                    # 如果 Port 全部被佔用，退回原本的方法
                    video_url = f"/app/static/output/{urllib.parse.quote(target_file)}"
                has_video = True
                if os.path.exists(srt_path):
                    subs = parse_srt(srt_path)
                    subtitles_json = json.dumps(subs)
            else:
                #檔案遺失處理
                st.error(f"檔案不存在: {target_file}")
                st.session_state.current_playing_name = None
                st.rerun()

        #左欄:影片播放器
        with col_player:
            if has_video:
                 #這裡呼叫拆分後的播放器HTML(僅含影片標籤)
                self.render_video_player_only(video_url, subtitles_json)
            else:
                st.empty()
                st.markdown(
                    """
                    <div style="
                        border: 2px dashed #ccc; 
                        border-radius: 10px; 
                        height: 400px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        color: #888;
                        background-color: #f9f9f9;
                    ">
                        <h3>請從左側選擇影片播放</h3>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

        #右欄:歌詞列表
        with col_lyrics:
            if has_video:
                pass 
            else:
                st.info("暫無歌詞資料")

        #渲染整合後的HTML
        #因為iframe限制，我們必須把「影片」和「歌詞」寫在同一個components.html裡
        #才能透過JS互相控制(點歌詞跳轉影片)
        #我們使用CSS來模擬Streamlit的左右分欄效果
        if has_video:
            self.render_split_layout_player(video_url, subtitles_json,target_file)

        if target_file:
            st.markdown("---")
            st.subheader("為您推薦")
            
            #從檔名中提取出歌曲名稱來當作搜尋關鍵字(去除副檔名)
            #視你的檔案命名規則而定，你也可以在這裡加入額外的字串處理
            song_name = os.path.splitext(target_file)[0] 
            
            with st.spinner("正在尋找推薦歌曲..."):
                #呼叫你的推薦函式
                recommendations = get_cached_recommendations(song_name, count=3)
            
            if recommendations:
                def add_to_queue_callback(url, title, semi, lang, use_gem, api_key):
                    st.session_state.url_input_key = url
                    
                    #執行排程邏輯
                    if not task_manager.is_processing:
                        success = task_manager.start_task(url, self.output_dir, semi, lang, use_gem, api_key)
                        if success:
                            st.toast(f"已將《{title}》加入排程！")
                        else:
                            st.error("任務啟動失敗。")
                    else:
                        st.warning("目前已有任務正在執行，請稍候。")
                #設定每排最多顯示幾張卡片
                max_cols_per_row = 4
                
                #將推薦清單分組，處理超過設定數量的情況
                for i in range(0, len(recommendations), max_cols_per_row):
                    chunk = recommendations[i:i + max_cols_per_row]
                    cols = st.columns(len(chunk))
                    
                    for col, rec in zip(cols, chunk):
                        with col:
                            with st.container(border=True):
                                #利用videoId取得YouTube影片縮圖
                                thumbnail_url = f"https://img.youtube.com/vi/{rec['videoId']}/mqdefault.jpg"
                                st.image(thumbnail_url, width='stretch')
                                
                                #顯示標題與歌手
                                st.markdown(f"**{rec['title']}**")
                                artist_name = rec['artist']
                                real_photo_url = rec['artist_img']
                                if real_photo_url.startswith("//"):
                                    real_photo_url = "https:" + real_photo_url

                                artist_html = f"""
                                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                                    <img src="{real_photo_url}" width="24" height="24" 
                                         style="border-radius: 50%; object-fit: cover;" 
                                         referrerpolicy="no-referrer" 
                                         onerror="this.src='https://ui-avatars.com/api/?name={artist_name}&background=random&color=fff&rounded=true&size=24'">
                                    <span style="font-size: 14px; color: #555;">{artist_name}</span>
                                </div>
                                """
                                st.markdown(artist_html, unsafe_allow_html=True)
                                
                                yt_url = f"https://www.youtube.com/watch?v={rec['videoId']}"
                                
                                lang_mapping = {"中文": "zh", "英文": "en", "閩南語": "nan"}
                                card_lang_key = st.selectbox(
                                    "辨識語言", 
                                    options=list(lang_mapping.keys()), 
                                    key=f"lang_sel_{rec['videoId']}",
                                    label_visibility="collapsed"
                                )
                                card_lang_val = lang_mapping[card_lang_key]
                                #建立左右兩個按鈕的區塊
                                btn_col1, btn_col2 = st.columns(2)
                                
                                #左邊按鈕:加入處理佇列
                                with btn_col1:
                                    use_gem = st.session_state.get("use_gemini_chk", True)
                                    api_key_val = st.session_state.get("gemini_api_key", "").strip()
                                    st.button(
                                        "➕ 加入", 
                                        key=f"btn_add_{rec['videoId']}", 
                                        use_container_width=True,
                                        on_click=add_to_queue_callback, 
                                        args=(yt_url, rec['title'], semitones, card_lang_val, use_gem, api_key_val) 
                                    )

                                #右邊按鈕:複製連結
                                with btn_col2:
                                    btn_id = f"copy-btn-{rec['videoId']}"
                                    
                                    copy_button_html = f"""
                                    <button id="{btn_id}" onclick="copyUrl_{rec['videoId']}()" 
                                            style="width: 100%; padding: 6px 12px; background-color: transparent; 
                                                   color: #31333F; border: 1px solid rgba(49, 51, 63, 0.2); 
                                                   border-radius: 8px; cursor: pointer; font-family: sans-serif; 
                                                   font-size: 14px; transition: all 0.3s; margin-top: 1px;">
                                        📋 複製
                                    </button>
                                    <script>
                                    function copyUrl_{rec['videoId']}() {{
                                        navigator.clipboard.writeText('{yt_url}').then(() => {{
                                            var btn = document.getElementById('{btn_id}');
                                            btn.innerText = '✅ 已複製';
                                            btn.style.borderColor = '#00CC66';
                                            btn.style.color = '#00CC66';
                                            setTimeout(() => {{
                                                btn.innerText = '📋 複製';
                                                btn.style.borderColor = 'rgba(49, 51, 63, 0.2)';
                                                btn.style.color = '#31333F';
                                            }}, 2000);
                                        }}).catch(err => {{
                                            console.error('複製失敗: ', err);
                                        }});
                                    }}
                                    </script>
                                    """
                                    components.html(copy_button_html, height=45)
            else:
                st.info("目前沒有推薦歌曲。")

        # 在所有表單與操作處理完畢後，再渲染側邊欄，確保狀態是最新的
        with sidebar_placeholder.container():
            self.sidebar_queue_fragment()

    #已棄用
    def render_video_player_only(self, video_url, subtitles_json):
        """
        這個函式已經被棄用，改用 render_split_layout_player。
        因為 Streamlit Component 是 iframe,跨 iframe 無法通訊
        """
        pass
    
    #實際渲染函式
    def render_split_layout_player(self, video_url, subtitles_json,target_file):
        """
        渲染一個包含「左邊影片」和「右邊歌詞」的HTML Component。
        修改為響應式高度設計。
        """
        html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    /* Reset & Base */
                body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; padding: 0; overflow: hidden; }}
                
                /* 主要容器：使用 Flexbox 佈局 */
                .main-container {{
                    display: flex;
                    flex-direction: row;
                    width: 100%;
                    /* 設定一個最大高度，避免在超寬螢幕上歌詞區太矮，
                       使用 vh (viewport height) 讓它稍微有點彈性，但這裡主要靠 aspect-ratio */
                    height: 100vh; 
                    max-height: 650px; /* 限制最大高度，避免佔據太多垂直空間 */
                    gap: 15px;
                }}
                
                /* 左邊：影片區 */
                .video-section {{
                    flex: 7; /* 影片佔 60% 左右 */
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start; /* 靠上對齊 */
                }}
                
                .video-wrapper {{
                    position: relative;
                    width: 100%;
                    background: #000; 
                    border-radius: 8px; 
                    overflow: hidden; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                    
                    /* 關鍵：使用 aspect-ratio 自動維持 16:9 比例 */
                    aspect-ratio: 16 / 9;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}

                video {{
                    width: 100%;
                    height: 100%;
                    object-fit: contain; /* 確保影片完整顯示不變形 */
                }}
                
                .time-display {{
                    text-align: center; 
                    font-family: monospace; 
                    font-size: 14px; 
                    color: #555;
                    margin-top: 8px;
                }}

                /* 右邊：歌詞區 */
                .lyrics-section {{
                    flex: 4; /* 歌詞佔 40% */
                    display: flex;
                    flex-direction: column;
                    /* 讓歌詞區高度跟隨影片區的高度 (或是外層容器高度) */
                    height: 85%; 
                    border: 1px solid #eee;
                    border-radius: 12px;
                    background: #fff;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                    overflow: hidden; /* 防止撐開 */
                }}

                .lyrics-header {{
                    padding: 12px 15px;
                    border-bottom: 1px solid #eee;
                    font-weight: bold;
                    font-size: 16px;
                    background: #f8f9fa;
                    color: #333;
                }}

                .lyrics-body {{
                    flex-grow: 1;
                    overflow-y: auto;
                    padding: 15px;
                    scroll-behavior: smooth;
                }}

                /* 歌詞樣式 */
                .lyric-line {{
                    margin-bottom: 16px;
                    font-size: 20px;
                    line-height: 1.5;
                    color: #bbb;
                    transition: all 0.2s ease;
                    cursor: pointer;
                    padding: 6px 10px;
                    border-radius: 6px;
                }}
                
                .lyric-line:hover {{
                    background-color: #f5f5f5;
                    color: #666;
                }}

                .active-line {{
                    color: #000 !important;
                    font-weight: bold;
                    font-size: 21px;
                    background: #e6f3ff; 
                    border-left: 4px solid #2196F3;
                }}
                
                /* 響應式處理：當寬度變窄時 (例如手機或分割視窗) */
                @media (max-width: 800px) {{
                    .main-container {{
                        flex-direction: column;
                        max-height: none; /* 取消高度限制 */
                        height: auto;
                    }}
                    .video-section {{
                        flex: none;
                        width: 100%;
                    }}
                    .lyrics-section {{
                        flex: none;
                        width: 50%;
                        height: 300px; /* 手機版給歌詞區固定高度 */
                    }}
                }}
            </style>

            <div class="main-container">
                <div class="video-section">
                    <div class="video-wrapper">
                        <video id="myVideo" controls autoplay playsinline crossorigin="anonymous">
                            <source src="{video_url}" type="video/mp4">
                        </video>
                    </div>
                    <div class="time-display">
                        <span id="timeDisplay">0.000</span> s
                    </div>
                </div>

                <div class="lyrics-section">
                    <div class="lyrics-header">{target_file}</div>
                    <div class="lyrics-body" id="subtitleContent"></div>
                </div>
            </div>

            <script>
                var vid = document.getElementById("myVideo");
                var display = document.getElementById("timeDisplay");
                var subContent = document.getElementById("subtitleContent");
                var subtitles = {subtitles_json};
                
                if (subtitles.length > 0) {{
                    subtitles.forEach((sub, index) => {{
                        var p = document.createElement("div");
                        p.className = "lyric-line";
                        p.id = "sub-" + index;
                        p.innerHTML = sub.text;
                        p.onclick = function() {{
                            vid.currentTime = sub.start;
                            vid.play();
                        }};
                        subContent.appendChild(p);
                    }});
                }} else {{
                    subContent.innerHTML = "<p style='color:#999; text-align:center; margin-top:40px;'>無字幕資料</p>";
                }}

                var activeIndex = -1;
                vid.ontimeupdate = function() {{
                    var t = vid.currentTime;
                    display.innerText = t.toFixed(3);
                    var currentIdx = -1;
                    for (var i = 0; i < subtitles.length; i++) {{
                        if (t >= subtitles[i].start && t < subtitles[i].end + 0.2) {{
                            currentIdx = i;
                            break;
                        }}
                    }}
                    if (currentIdx !== -1 && currentIdx !== activeIndex) {{
                        if (activeIndex !== -1) {{
                            var prev = document.getElementById("sub-" + activeIndex);
                            if (prev) prev.classList.remove("active-line");
                        }}
                        var curr = document.getElementById("sub-" + currentIdx);
                        if (curr) {{
                            curr.classList.add("active-line");
                            curr.scrollIntoView({{behavior: "smooth", block: "center"}});
                        }}
                        activeIndex = currentIdx;
                    }}
                }};
            </script>
            </body>
            </html>
        """
        
        # 將 HTML 寫入靜態檔案，並透過 iframe 載入
        # 這樣在 st.rerun() 時，只要 URL 不變，Streamlit 就不會重新載入 iframe，影片就不會中斷！
        import re
        safe_name = os.path.splitext(target_file)[0]
        safe_name = re.sub(r'[\\/*?:"<>|]', "", safe_name)
        player_filename = f"player_{safe_name}.html"
        player_path = os.path.join(self.output_dir, player_filename)
        
        with open(player_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        local_port = get_local_server_port()
        import urllib.parse
        if local_port:
            host_ip = get_local_ip()
            player_url = f"http://{host_ip}:{local_port}/{urllib.parse.quote(player_filename)}"
        else:
            player_url = f"/{self.output_dir}/{urllib.parse.quote(player_filename)}"
            
        components.iframe(player_url, height=600, scrolling=False)

if __name__ == "__main__":
    app = WebDisplay()
    app.render()