"""

Resizes inernal and external data and moves it to same directory

"""

import argparse
from multiprocessing import Pool
from pathlib import Path
import os

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm



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

def resize_and_move_image(args):
    image_name, size, source_path, target_path = args
    image_file = os.path.join(source_path, (image_name+'.png'))
    #print(image_file)
    if os.path.exists(image_file):
        # create the size by size RGB image
        im = Image.open(image_file)
        
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
        pool.map(resize_and_move_image, batch);
