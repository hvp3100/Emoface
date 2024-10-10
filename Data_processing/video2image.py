import os
import argparse
from multiprocessing import Pool

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path

def convert(video_path):
    video_name = video_path.split('/')[-1]
    output_folder = "/media/admin123/T71/BD_p"
    os.makedirs(output_folder, exist_ok=True)

    os.system(f'ffmpeg -i {video_path} -f image2 -r 20 -b:v 5626k {output_folder}/{video_name.split(".")[0]}_%05d.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str, default='/media/admin123/T7/BD')
    args = parser.parse_args()
    video_paths = findAllFile(args.video_folder)
    pool = Pool(processes=8)
    pool.map(convert, video_paths)
    pool.close()
    pool.join()
