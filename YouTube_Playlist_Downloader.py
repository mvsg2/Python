from pytube import Playlist

def YouTubePlaylistDownloader(playlist_url, download_path):

    playlist = Playlist(url=playlist_url)
    print()
    print("Playlist title:", playlist.title)
    print("Number of videos:", len(playlist.video_urls))
    print("Downloading the playist...\n")
    video_count = 0
    for i, video in zip(range(len(playlist.video_urls)), playlist.videos):
        print(i+1, end=' ')
        print("Downloading -- {}".format(video.title))
        video.streams.get_highest_resolution().download(output_path=download_path)
        print("Downloaded the video - {}".format(video.title))
        video_count += 1
    if (video_count == len(playlist.video_urls)):
        print("\nDownloaded all videos of the playlist!")
    else:
        print("Downloaded {} videos of the playlist".format(video_count))

if __name__ == "__main__":
    YouTubePlaylistDownloader(playlist_url="https://www.youtube.com/playlist?list=PLWcODnEVd5uvHS8JueTHXzZymxGYbYL8t", download_path="E:/YouTube_Downloads/Gulliver's Travels/")
