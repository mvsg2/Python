from pytube import YouTube 
from sys import argv

download_folder = 'E:/Quantum Computing/Qubit by Qubit (QxQ)/The Coding School and IBM Quantum course on Quantum Computing/Semester-2 - Spring 2023/Lecture Videos/'
same_title = False
title = ''

title = input('Enter desired title: ')

links_file = open('YouTube_links.txt')
links = []
for i in links_file:
    links.append(i[:-1])
print('Total videos: {}'.format(len(links)))
print('Downloading the video...')

for i in range(len(links)):
    link = links[i]
    yt = YouTube(link)

    #yd2 = yt.streams.filter(res="720p")
    yd2 = yt.streams.get_highest_resolution()

    #download_name = '{}-{}.mp4'.format(title, i+1)
    download_name = '{}.mp4'.format(title)

    #print('Downloading Video-{}...'.format(i+1), end=' ')

    # yd2[0].download(output_path=download_folder, filename=download_name)
    yd2.download(output_path=download_folder, filename=download_name)

    print('-- Downloaded')