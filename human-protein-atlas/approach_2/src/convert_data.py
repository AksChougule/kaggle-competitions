"""
Based on rebryk's script from here:
https://github.com/rebryk/kaggle/tree/master/human-protein/scripts

Converts already downloaded external data to usable image files

"""

import argparse
from multiprocessing import Pool
from pathlib import Path
import os

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

#COLORS = ['red', 'green', 'blue', 'yellow']
channels = ['_yellow','_red', '_green', '_blue']


def get_config():
    parser = argparse.ArgumentParser('Convert external data for Human Protein Atlas Image Classification Challenge')
    parser.add_argument('--csv', type=str, required=True, help='path to csv file')
    parser.add_argument('--size', type=int, required=True, help='new image size')
    parser.add_argument('--src', type=str, required=True, help='path to folder with images')
    parser.add_argument('--dest', type=str, default=None, help='path to folder with converted images')
    parser.add_argument('--n_threads', type=int, default=4, help='number of threads')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    return parser.parse_args()


def get_batches(values, batch_size):
    for start in range(0, len(values), batch_size):
        yield values[start: start + batch_size]

def create_rgb_image(channels: list, size) -> np.ndarray:
    rgb_image = np.zeros(shape=(size, size, 3), dtype=np.float)
    
    if os.path.exists(channels[0]):
        yellow = Image.open(channels[0])
        im_y= np.asarray(yellow.resize((size,size), Image.ANTIALIAS))
        # yellow is red + green
        rgb_image[:, :, 0] += im_y[:,:,0]/2   
        rgb_image[:, :, 1] += im_y[:,:,0]/2
    
    # loop for R,G and B channels
    for index, channel in enumerate(channels[1:]):
        im = Image.open(channel)
        img = im.resize((size,size), Image.ANTIALIAS)
        current_image = np.asarray(img)
        if len(current_image.shape) != 3:
            break
        rgb_image[:, :, index] += current_image[:,:,index]
    # Normalize image
    rgb_image = (rgb_image / rgb_image.max()) * 255
    return rgb_image.astype(np.uint8)

def convert_sample(args):
    #name, size, src, dest = args
    image_name, size, source_path, target_path = args
    image_names = [os.path.join(source_path, image_name) + x + '.jpg' for x in channels]
    
    if os.path.exists(image_names[0]) and os.path.exists(image_names[1]) and os.path.exists(image_names[2]) and os.path.exists(image_names[3]):
        # create the size by size RGB image
        rgb_image = create_rgb_image(image_names, size)       
        im = Image.fromarray(rgb_image)    

        # resize to the defined size
        im = im.resize((size, size)) 

        # save the resized RGB image
        image_name = image_name + '.png'
        new_image = os.path.join(target_path,image_name)
        im.save(new_image)   

if __name__ == '__main__':
    config = get_config()

    df = pd.read_csv(config.csv)
    print(f'#images: {len(df)}')

    tasks = [(it, config.size, config.src, config.dest or config.src) for it in df.Id.values]
    batches = list(get_batches(tasks, batch_size=config.batch_size))

    pool = Pool(config.n_threads)

    for batch in tqdm(batches):
        pool.map(convert_sample, batch);
