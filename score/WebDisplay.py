import streamlit as st
import streamlit.components.v1 as components
import os
import base64
import glob
import shutil
import time
import threading
import re
import json
from datetime import datetime

#--- 1. 模擬環境與模組匯入 ---
try:
    from YouTubeDownload import YouTubeDownload
    from MusicTools import MusicTools
except ImportError as e:
    print(f"模組匯入警告: {e}")

#--- 2. 核心邏輯與工具函式 ---

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
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

#--- 3. 背景任務管理 (Singleton 模式) ---
@st.cache_resource
class TaskManager:
    def __init__(self):
        self.is_processing = False
        self.current_task_url = None
        self.last_completed_file = None
        self.status_message = ""
        self.lock = threading.Lock()

    #輸入網址開始執行
    def start_task(self, url, output_dir, pitch_steps=0):
        with self.lock:
            if self.is_processing:
                return False
            
            self.is_processing = True
            self.current_task_url = url
            self.status_message = "啟動處理程序..."
            
            thread = threading.Thread(target=self._worker, args=(url, output_dir, pitch_steps))
            thread.daemon = True
            thread.start()
            return True
    
    #開始執行的具體步驟
    def _worker(self, url, output_dir, pitch_steps):
        if (not(YouTubeDownload.check_YouTube_available(url))):
            self.status_message = "連結輸入錯誤或網路不可用"
            return
        try:
            self.status_message = "下載 YouTube 音訊與視訊..."
            audio = YouTubeDownload.download_youtube_audio(url)
            video = YouTubeDownload.download_youtube_video(url)
            
            self.status_message = "進行人聲分離..."
            separation = MusicTools.run_separation(audio)

            target_audio = separation[0] 
            
            #--- 變調處理 ---
            if pitch_steps != 0:
                self.status_message = f"進行升降調處理({pitch_steps}半音)..."
                
                #定義變調後的檔名
                path_parts = os.path.splitext(target_audio)
                shifted_audio_path = f"{path_parts[0]}_pitch{pitch_steps}{path_parts[1]}"
                
                #FFmpeg變調
                target_audio = MusicTools.change_pitch_ffmpeg(target_audio, pitch_steps)
            #-----------------------------------

            self.status_message = "合併影片與音訊..."

            #使用變調後的音訊進行合併
            merge = YouTubeDownload.merge_video_audio(video, target_audio)

            self.status_message = "Whisper 歌詞辨識中..."
            AsrReturn = MusicTools.run_ASR(separation[1])
            
            srt_source_path = None
            if AsrReturn and len(AsrReturn) > 0:
                srt_source_path = AsrReturn[0]
                #srt_source_path = r"H:\NFU\score\download\測試.srt" #測試用路徑

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
            
            #最終檔名
            out_video_path = os.path.join(output_dir, safe_name)
            out_srt_path = os.path.join(output_dir, f"{final_name_base}.srt")
            
            if os.path.exists(merge):
                if os.path.abspath(merge) != os.path.abspath(out_video_path):
                    if os.path.exists(out_video_path): os.remove(out_video_path)
                    os.replace(merge, out_video_path)
            else:
                with open(out_video_path, 'w') as f: f.write("dummy content")

            if srt_source_path and os.path.exists(srt_source_path):
                if os.path.exists(out_srt_path): os.remove(out_srt_path)
                shutil.move(srt_source_path, out_srt_path)

            self.last_completed_file = safe_name
            self.status_message = f"✓ 完成：{safe_name}"
            
        except Exception as e:
            self.status_message = f"✗ 錯誤: {str(e)}"
            print(f"Worker Error: {e}")
        finally:
            with self.lock:
                self.is_processing = False

task_manager = TaskManager()

#--- 4. 介面顯示類別 ---

class WebDisplay:
    #初始化
    def __init__(self):
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        if 'is_initialized' not in st.session_state:
            st.session_state.is_initialized = True
            st.session_state.current_playing_name = None

    #左方佇列
    @st.fragment(run_every=2)
    def sidebar_queue_fragment(self):
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

                if st.button(label, key=f"btn_{filename}", use_container_width=True):
                    st.session_state.current_playing_name = filename
                    st.rerun()
        else:
            st.caption("尚無檔案，請新增任務。")
        
        st.markdown("---")
        if st.button("清除所有紀錄", use_container_width=True):
            clear_output_folder(self.output_dir)
            st.session_state.current_playing_name = None
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

        #--- 側邊欄 ---
        with st.sidebar:
            self.sidebar_queue_fragment()

        #--- 輸入框 ---
        #建立一個容器放在最上方
        with st.container():
            #使用 columns 稍微限縮寬度，或者直接全寬
            #這裡使用全寬讓輸入框更明顯
            with st.form("task_form"):
                c1, c2, c3 = st.columns([6, 1, 2], vertical_alignment="bottom") # 調整欄位比例
                with c1:
                    url_input = st.text_input("YouTube URL", placeholder="請輸入 YouTube 連結...", label_visibility="collapsed")
                with c2:
                    # 新增：升降調選擇 (-6 到 +6 半音)
                    semitones = st.number_input("升降調 (半音)", min_value=-12, max_value=12, value=0, step=1, help="正數為升調，負數為降調")
                with c3:
                    submitted = st.form_submit_button("加入背景處理", type="primary", use_container_width=True)
                
                if submitted:
                    if url_input:
                        if not task_manager.is_processing:
                            success = task_manager.start_task(url_input, self.output_dir, semitones)
                            if success:
                                st.toast(f"已加入排程(變調:{semitones})，請留意側邊欄狀態！")
                            else:
                                st.warning("任務啟動失敗。")
                        else:
                            st.warning("目前已有任務正在執行，請稍候。")
                    else:
                        st.warning("請輸入有效的 URL")

        st.markdown("---") #分隔線

        #--- 分欄:左播放器，右歌詞 ---
        #調整比例
        col_player, col_lyrics = st.columns([1.6, 1]) 

        target_file = st.session_state.current_playing_name
        
        #預先處理好路徑和資料
        video_b64 = ""
        subtitles_json = "[]"
        has_video = False

        if target_file:
            video_path = os.path.join(self.output_dir, target_file)
            srt_path = os.path.splitext(video_path)[0] + ".srt"
            
            if os.path.exists(video_path):
                video_b64 = get_video_base64(video_path)
                has_video = True
                if os.path.exists(srt_path):
                    subs = parse_srt(srt_path)
                    subtitles_json = json.dumps(subs)
            else:
                #檔案遺失處理
                st.error(f"檔案不存在: {target_file}")
                st.session_state.current_playing_name = None
                st.rerun()

        #--- 左欄:影片播放器 ---
        with col_player:
            if has_video:
                 #這裡呼叫拆分後的播放器 HTML (僅含影片標籤)
                self.render_video_player_only(video_b64, subtitles_json)
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

        #--- 右欄:歌詞列表 ---
        with col_lyrics:
            if has_video:
                pass 
            else:
                st.info("暫無歌詞資料")

        #--- 渲染整合後的HTML ---
        #因為iframe限制，我們必須把「影片」和「歌詞」寫在同一個components.html裡
        #才能透過JS互相控制(點歌詞跳轉影片)
        #我們使用CSS來模擬Streamlit的左右分欄效果
        if has_video:
            #我們把剛剛建立的col_player和col_lyrics內容清空或當作佔位符
            #直接在下方渲染一個全寬的Component，內部自己分左右
            #為了版面好看，我們可以把上面的st.subheader移除，直接寫在HTML裡
            
            #清除上面兩個col的暫位內容(視覺上)-實際上Streamlit無法動態刪除已渲染的元件
            #所以上面的col_player/lyrics只是用來顯示標題或空狀態
            #當有影片時，我們改用下面這個全版元件來取代原本的分欄顯示
            
            self.render_split_layout_player(video_b64, subtitles_json,target_file)

    #已棄用
    def render_video_player_only(self, video_b64, subtitles_json):
        """
        這個函式已經被棄用，改用 render_split_layout_player。
        因為 Streamlit Component 是 iframe,跨 iframe 無法通訊
        """
        pass
    
    #實際渲染函式
    def render_split_layout_player(self, video_b64, subtitles_json,target_file):
        """
        渲染一個包含「左邊影片」和「右邊歌詞」的HTML Component。
        修改為響應式高度設計。
        """
        
        html_content = f"""
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
                        <video id="myVideo" controls autoplay playsinline>
                            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
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
        """
        
        #Streamlit的components.html需要一個初始高度
        #雖然我們在CSS裡已經設為響應式，但這個Python參數決定了iframe挖多大的洞
        #設定600~700左右通常能適配大部分16:9影片在寬螢幕下的高度
        components.html(html_content, height=600, scrolling=False)

if __name__ == "__main__":
    app = WebDisplay()
    app.render()