import gradio as gr
import os
from MusicTools import MusicSeparation
from YouTubeDownload import YouTubeDownload

def fun(url1):
    audio = YouTubeDownload.download_youtube_audio(url1)
    video = YouTubeDownload.download_youtube_video(url1)
    separtionMusic = MusicSeparation.run_separation(audio)
    merge = YouTubeDownload.merge_video_audio(video_path=video, audio_path=separtionMusic[0])

    # 去掉多餘的引號 & 替換不合法字元
    merge = merge.strip('"').strip("'")
    safe_name = os.path.basename(merge).replace("／", "_").replace("/", "_").replace("\\", "_")
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, safe_name)

    if merge != out_path:
        try:
            os.replace(merge, out_path)  # 搬移檔案
        except Exception as e:
            print("搬檔失敗:", e)
            out_path = merge  # 如果搬失敗，就還是用原本路徑

    if not os.path.exists(out_path):
        return f"檔案不存在: {out_path}"

    return out_path  # 給 gr.Video()


iface = gr.Interface(
    fn=fun,
    inputs=gr.Textbox(lines=1, placeholder="請輸入YouTube連結"),
    outputs=gr.Video(),
    title="哈哈標題",
    description="輸入Youtube連結自動下載並分離人聲及音樂",
    share=True,
)

os.system("start http://127.0.0.1:7860")

iface.launch()
