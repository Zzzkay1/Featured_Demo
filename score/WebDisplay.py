import streamlit as st
import streamlit.components.v1 as components
import os
import base64

# 嘗試匯入您的模組
try:
    from YouTubeDownload import YouTubeDownload
    from MusicSeparation import MusicSeparation
except ImportError:
    st.error("找不到 YouTubeDownload 或 MusicSeparation 模組，請檢查檔案位置。")

def get_video_base64(video_path):
    """讀取影片並轉為 Base64 字串"""
    with open(video_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

class WebDisplay:
    def __init__(self):
        if 'result_text' not in st.session_state:
            st.session_state.result_text = ""
        if 'video_path' not in st.session_state:
            st.session_state.video_path = None

    def render(self):
        st.title("AI 影片處理與時間碼播放器")
        
        col_input, col_output = st.columns([1, 1]) 

        # --- 左側：輸入區 ---
        with col_input:
            url_input = st.text_input("YouTube URL", placeholder="請輸入YouTube連結")
            
            if st.button("Submit", type="primary", use_container_width=True):
                if url_input:
                    try:
                        with st.spinner("正在下載與處理中... (請稍候)"):
                            # === 這裡執行您的核心邏輯 ===
                            audio = YouTubeDownload.download_youtube_audio(url_input)
                            video = YouTubeDownload.download_youtube_video(url_input)
                            separation = MusicSeparation.run_separation(audio)
                            merge = YouTubeDownload.merge_video_audio(video, separation[0])

                            # 檔案搬移邏輯
                            merge = merge.strip('"').strip("'")
                            safe_name = os.path.basename(merge).replace("／", "_").replace("/", "_").replace("\\", "_")
                            output_dir = "output"
                            os.makedirs(output_dir, exist_ok=True)
                            out_path = os.path.join(output_dir, safe_name)
                            
                            if os.path.abspath(merge) != os.path.abspath(out_path):
                                if os.path.exists(out_path): os.remove(out_path)
                                os.replace(merge, out_path)
                            else:
                                out_path = merge
                            
                            # 更新狀態
                            st.session_state.result_text = "處理完成！"
                            st.session_state.video_path = out_path
                            st.rerun()

                    except Exception as e:
                        st.error(f"錯誤: {e}")
                else:
                    st.warning("請輸入連結")

        # --- 右側：輸出區 (含時間碼) ---
        with col_output:
            st.caption("Output Result")
            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                st.success(st.session_state.result_text)
                
                # 1. 取得影片 Base64
                video_b64 = get_video_base64(st.session_state.video_path)
                
                # 2. 定義 HTML/JS 播放器
                # 注意下面的 <div id="timeDisplay"> 就是時間碼出現的地方
                video_html = f"""
                    <div style="background-color: black; padding: 10px; border-radius: 8px;">
                        <video id="myVideo" width="100%" controls autoplay>
                            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                        </video>
                        
                        <div style="margin-top: 10px; color: white; font-family: monospace; text-align: center; font-size: 18px;">
                            ⏱️ 目前時間: <span id="timeDisplay" style="color: #ff4b4b; font-weight: bold;">0.00</span> 秒
                        </div>
                    </div>

                    <script>
                        var vid = document.getElementById("myVideo");
                        var display = document.getElementById("timeDisplay");
                        
                        // 當影片播放位置改變時，更新文字
                        vid.ontimeupdate = function() {{
                            display.innerText = vid.currentTime.toFixed(3); 
                        }};
                    </script>
                """
                components.html(video_html, height=450)
            else:
                # 空白狀態
                st.info("請在左側輸入連結並送出")

if __name__ == "__main__":
    app = WebDisplay()
    app.render()