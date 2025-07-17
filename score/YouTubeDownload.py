from bs4 import BeautifulSoup
from pytubefix import YouTube
import os
import subprocess
import re
class YouTubeDownload:
    def __init__(self):
        pass
    @staticmethod
    def sanitize_filename(name):
        #移除檔名中不合法的字元
        return re.sub(r'[\\/:*?"<>|]', '_', name)
    @staticmethod
    def download_youtube_audio(video_url):
        try:
            yt = YouTube(video_url)
            title = YouTubeDownload.sanitize_filename(yt.title)
            audio_stream = yt.streams.get_audio_only()

            os.makedirs("download", exist_ok=True)
            os.path.join("download", f"{title}.m4a")
            m4a_path = audio_stream.download(output_path="download", filename=f"{title}.m4a")

            return f'"{m4a_path}"'
        except Exception as e:
            return f"錯誤：{e}"
    @staticmethod
    def download_youtube_video(video_url):
        try:
            yt = YouTube(video_url)
            title = YouTubeDownload.sanitize_filename(yt.title)

            #優先取得 1080p 無聲影片
            video_stream = yt.streams.filter(res="1080p", mime_type="video/mp4", only_video=True).first()
            #若無1080p，退而求其次找最高畫質的影片
            if not video_stream:
                video_stream = yt.streams.filter(mime_type="video/mp4").order_by("resolution").desc().first()
            if not video_stream:
                return "找不到合適的視訊串流"

            os.makedirs("download", exist_ok=True)
            os.path.join("download", f"{title}.mp4")
            video_path = video_stream.download(output_path="download", filename=f"{title}.mp4")

            return f'"{video_path}"'
        except Exception as e:
            return f"錯誤：{e}"

    def merge_video_audio(video_path, audio_path, output_path = ""):
        video_path = video_path.strip('"')
        audio_path = audio_path.strip('"')
        output_path = output_path.strip('"')
        if output_path == "" :
            output_path = f"{video_path.rsplit('.', 1)[0]}_merged.mp4"
        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-strict", "experimental",
                output_path,
                "-y"
            ]
            subprocess.run(cmd, check=True)
            return f'"{output_path}"'
        except Exception as e:
            return f"合併失敗：{e}"