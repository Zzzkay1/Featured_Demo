from bs4 import BeautifulSoup
from pytubefix import YouTube
import pytubefix
import os
import subprocess
import re
from pytubefix.exceptions import VideoUnavailable, RegexMatchError
from ytmusicapi import YTMusic

class YouTubeDownload:
    def __init__(self):
        pass
    @staticmethod
    def sanitize_filename(name):
        if name:
            try:
                # 修正 Windows 下常見的 pytubefix 亂碼問題 (UTF-8 被誤用 cp950 解碼)
                name = name.encode('cp950').decode('utf-8')
            except Exception:
                pass
        #移除檔名中不合法的字元
        return re.sub(r'[\\/:*?"<>|]', '_', name)

    @staticmethod
    def get_correct_youtube_title(url):
        import urllib.request
        import re
        import html as html_lib
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                html_content = response.read().decode('utf-8')
                match = re.search(r'<title>(.*?)</title>', html_content)
                if match:
                    title = match.group(1).replace(' - YouTube', '').strip()
                    title = html_lib.unescape(title)
                    invalid_titles = ["你的瀏覽器已不適用", "Your browser is no longer supported", "Before you continue", "YouTube"]
                    for invalid in invalid_titles:
                        if invalid in title:
                            return None
                    return title
        except Exception as e:
            print(f"Error fetching title directly: {e}")
        return None
    @staticmethod
    def get_actual_url(query: str) -> str:
        from pytubefix import Search
        if query.startswith("http://") or query.startswith("https://"):
            return query
        try:
            results = Search(query)
            if results.videos:
                return results.videos[0].watch_url
        except Exception as e:
            print(f"搜尋失敗: {e}")
        return None
    @staticmethod
    def check_YouTube_available(url: str) -> bool:
        try:
            #use_oauth=True 增加通過率，避免被當機器人
            #allow_oauth_cache=True 在本地儲存驗證資訊
            yt = YouTube(url)
            
            #檢查影片是否因版權、區域限制而無法播放
            yt.check_availability()
            
            #嘗試讀取標題，確認Metadata解析正常
            if yt.title:
                return True
                
        except VideoUnavailable:
            print(f"影片無法使用 (被下架或私人影片): {url}")
            return False
        except RegexMatchError:
            print(f"網址格式錯誤或解析失敗: {url}")
            return False
        except Exception as e:
            #捕捉其他非預期的錯誤
            print(f"檢查時發生未預期錯誤: {e}")
            return False
        print("未知錯誤")
        return False
    @staticmethod
    def download_youtube_audio(video_url:str|list, mode = "audio")->str|list:
        if mode == "audio" :
            if (YouTubeDownload.check_YouTube_available(video_url)==False):
                return None
            try:
                yt = YouTube(video_url)
                correct_title = YouTubeDownload.get_correct_youtube_title(video_url)
                title = YouTubeDownload.sanitize_filename(correct_title if correct_title else yt.title)
                
                audio_stream = yt.streams.filter(only_audio=True).first()

                os.makedirs("download", exist_ok=True)
                os.path.join("download", f"{title}.m4a")
                m4a_path = audio_stream.download(output_path="download", filename=f"{title}.m4a")

                return f'"{m4a_path}"'
            except Exception as e:
                raise e
        elif mode == "playlist":
            temp = []
            for audio in pytubefix.Playlist(video_url):
                if (YouTubeDownload.check_YouTube_available(audio)==False):
                    temp.append(None)
                    continue
                try:
                    yt = YouTube(audio)
                    title = YouTubeDownload.sanitize_filename(yt.title)
                    audio_stream = yt.streams.get_audio_only()

                    os.makedirs("download", exist_ok=True)
                    os.path.join("download", f"{title}.m4a")
                    temp.append(audio_stream.download(output_path="download", filename=f"{title}.m4a"))
                    
                except Exception as e:
                    raise e
            return temp
        
    @staticmethod
    def download_youtube_video(video_url:str|list, mode="video")->str|list|None:
        if mode == "video":
            if (YouTubeDownload.check_YouTube_available(video_url)==False):
                return None
            try:
                yt = YouTube(video_url)
                correct_title = YouTubeDownload.get_correct_youtube_title(video_url)
                title = YouTubeDownload.sanitize_filename(correct_title if correct_title else yt.title)

                #優先取得1080p無聲影片
                video_stream = yt.streams.filter(res="1080p", file_extension="mp4", progressive=False, type="video").first()
                #若無1080p，退而求其次找最高畫質的影片
                if not video_stream:
                    video_stream = yt.streams.filter(mime_type="video/mp4").order_by("resolution").desc().first()
                if not video_stream:
                    raise ValueError("找不到合適的視訊串流")

                os.makedirs("download", exist_ok=True)
                os.path.join("download", f"{title}.mp4")
                video_path = video_stream.download(output_path="download", filename=f"{title}.mp4")

                return f'"{video_path}"'
            except Exception as e:
                raise e
        elif mode == "playlist":
            temp = []
            for video in pytubefix.Playlist(video_url):
                if (YouTubeDownload.check_YouTube_available(video)==False):
                    temp.append(None)
                    continue
                try:
                    yt = YouTube(video)
                    title = YouTubeDownload.sanitize_filename(yt.title)

                    #優先取得1080p無聲影片
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
                    raise e
            return temp
    @staticmethod
    def get_YouTube_captions_languageCode(url:str)->list|None:
        if (YouTubeDownload.check_YouTube_available(url)):
            yt = YouTube(url)
            return list(yt.captions.keys())
        else:
            return None

    @staticmethod
    def download_captions_by_language_code(video_url:str|list, language_code:str|None)->str:
        try:
            yt = YouTube(video_url)
            correct_title = YouTubeDownload.get_correct_youtube_title(video_url)
            title = YouTubeDownload.sanitize_filename(correct_title if correct_title else yt.title)
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
            raise e


    def merge_video_audio(video_path:str, audio_path:str, output_path = "")->str:
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
                "-q:a", "0",
                "-y",
                output_path
                ]
            subprocess.run(cmd, check=True)
            return f'"{output_path}"'
        except Exception as e:
            raise Exception(f"合併失敗：{e}")

    def get_recommendations_by_artist(song_name, count=3):
            yt = YTMusic()
            search_results = yt.search(song_name, filter="songs")
            if not search_results:
                return None
            
            target_song = search_results[0]
            target_title = target_song['title']
            
            artists = target_song.get('artists', [])
            if not artists or 'id' not in artists[0] or not artists[0]['id']:
                return None
                
            artist_id = artists[0]['id']
            artist_page = yt.get_artist(artist_id)

            artist_img_url = "https://ui-avatars.com/api/?name=Artist&background=random" #預設圖片
            if 'thumbnails' in artist_page and artist_page['thumbnails']:
                artist_img_url = artist_page['thumbnails'][0]['url']

            all_songs = artist_page.get('songs', {}).get('results', [])
            
            recommendations = []
            for song in all_songs:
                if song['title'] != target_title and len(recommendations) < count:
                    recommendations.append({
                        'title': song['title'],
                        'videoId': song['videoId'],
                        'artist': artists[0]['name'],
                        'artist_img': artist_img_url
                    })
            return recommendations