from bs4 import BeautifulSoup
from pytubefix import YouTube
import pytubefix
import os
import subprocess
import re
def __init__():
    from .import YouTubeDownload

class YouTubeDownload:
    def __init__(self):
        pass
    @staticmethod
    def sanitize_filename(name):
        #移除檔名中不合法的字元
        return re.sub(r'[\\/:*?"<>|]', '_', name)
    @staticmethod
    def download_youtube_audio(video_url, mode = "audio"):
        if mode == "audio" :
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
        elif mode == "playlist":
            temp = []
            for audio in pytubefix.Playlist(video_url):
                try:
                    yt = YouTube(audio)
                    title = YouTubeDownload.sanitize_filename(yt.title)
                    audio_stream = yt.streams.get_audio_only()

                    os.makedirs("download", exist_ok=True)
                    os.path.join("download", f"{title}.m4a")
                    temp.append(audio_stream.download(output_path="download", filename=f"{title}.m4a"))
                    
                except Exception as e:
                    temp.append(f"錯誤：{e}")
            return temp
        
    @staticmethod
    def download_youtube_video(video_url, mode="video"):
        if mode == "video":
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
        elif mode == "playlist":
            temp = []
            for video in pytubefix.Playlist(video_url):
                try:
                    yt = YouTube(video)
                    title = YouTubeDownload.sanitize_filename(yt.title)

                    #優先取得 1080p 無聲影片
                    video_stream = yt.streams.filter(res="1080p", mime_type="video/mp4", only_video=True).first()
                    #若無1080p，退而求其次找最高畫質的影片
                    if not video_stream:
                        video_stream = yt.streams.filter(mime_type="video/mp4").order_by("resolution").desc().first()
                    if not video_stream:
                        temp.append("找不到合適的視訊串流")

                    os.makedirs("download", exist_ok=True)
                    os.path.join("download", f"{title}.mp4")
                    video_path = video_stream.download(output_path="download", filename=f"{title}.mp4")

                    temp.append(f'"{video_path}"')
                except Exception as e:
                    temp.append(f"錯誤：{e}")
            return temp
    

    @staticmethod
    def download_captions_by_language_code(video_url, language_code):
        try:
            yt = YouTube(video_url)
            title = YouTubeDownload.sanitize_filename(yt.title)
            caption = yt.captions
            #caption[language_code]

             #嘗試取得指定語言字幕
            if language_code in yt.captions:
                caption = yt.captions[language_code]
            elif f'a.{language_code}' in yt.captions:
                caption = yt.captions[f'a.{language_code}']
            else:
                available_languages = list(yt.captions.keys())
                return f"錯誤：此影片沒有 '{language_code}' 語言的字幕，可用語言如下：{available_languages}"

            #轉成srt格式
            srt_captions = caption.generate_srt_captions()

            #建立檔名並存檔
            filename = f"{title}_{language_code}.srt"
            with open(f"download/{filename}", "w", encoding="utf-8") as f:
                f.write(srt_captions)

            return os.path.join("/download",filename)

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
            "-q:a", "0",   # VBR: 0 = 最佳音質
            "-y",
            output_path
            ]
            subprocess.run(cmd, check=True)
            return f'"{output_path}"'
        except Exception as e:
            return f"合併失敗：{e}"