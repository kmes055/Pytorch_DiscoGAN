from __future__ import print_function
import os
from os.path import join, exists
import multiprocessing
import hashlib
import cv2
import sys
import zipfile
import argparse
from six.moves import urllib

def preprocess_facescrub(dirpath):
    data_dir = os.path.join(dirpath, 'facescrub')
    if os.path.exists(data_dir):
        print('Found Facescrub')
    else:
        os.mkdir(data_dir)
    files = ['./db/facescrub_actors.txt', './db/facescrub_actresses.txt']
    for f in files:
        with open(f, 'r') as fd:
            # strip first line
            fd.readline()
            names = []
            urls = []
            bboxes = []
            genders = []
            for line in fd.readlines():
                gender = f.split("_")[1].split(".")[0]
                components = line.split('\t')
                assert(len(components) == 6)
                name = components[0].replace(' ', '_')
                url = components[3]
                bbox = [int(_) for _ in components[4].split(',')]
                names.append(name)
                urls.append(url)
                bboxes.append(bbox)
                genders.append(gender)
        # every name gets a task
        last_name = names[0]
        task_names = []
        task_urls = []
        task_bboxes = []
        task_genders = []
        tasks = []
        for i in range(len(names)):
            if names[i] == last_name:
                task_names.append(names[i])
                task_urls.append(urls[i])
                task_bboxes.append(bboxes[i])
                task_genders.append(genders[i])
            else:
                tasks.append(
                    (data_dir, task_genders, task_names, task_urls, task_bboxes))
                task_names = [names[i]]
                task_urls = [urls[i]]
                task_bboxes = [bboxes[i]]
                task_genders = [genders[i]]
                last_name = names[i]
        tasks.append(
            (data_dir, task_genders, task_names, task_urls, task_bboxes))

        pool_size = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=pool_size, maxtasksperchild=2)
        pool.map(download_facescrub, tasks)
        pool.close()
        pool.join()


def download_facescrub(data_tuple):
    (data_dir, genders, names, urls, bboxes) = data_tuple

    assert(len(names) == len(urls))
    assert(len(names) == len(bboxes))
    # download using external wget
    CMD = 'wget -c -t 1 -T 3 "%s" -O "%s"'
    for i in range(len(names)):
        directory = join(data_dir, genders[i])

        if not exists(directory):
            print(directory)
            os.mkdir(directory)
        fname = hashlib.sha1(urls[i].encode('utf-8')).hexdigest() + "_" + names[i] + '.jpg'
        dst = join(directory, fname)
        print("downloading", dst)
        if exists(dst):
            print("already downloaded, skipping...")
            continue
        else:
            res = os.system(CMD % (urls[i], dst))
        # get face
        face_directory = join(directory, 'face')
        if not exists(face_directory):
            os.mkdir(face_directory)
        img = cv2.imread(dst)
        if img is None:
            # no image data
            os.remove(dst)
        else:
            face_path = join(face_directory, fname)
            face = img[bboxes[i][1]:bboxes[i][3], bboxes[i][0]:bboxes[i][2]]
            cv2.imwrite(face_path, face)
            #write bbox to file
            with open(join(directory, '_bboxes.txt'), 'a') as fd:
                bbox_str = ','.join([str(_) for _ in bboxes[i]])
                fd.write('%s %s\n' % (fname, bbox_str))


if __name__ == '__main__':
    preprocess_facescrub('./datasets/')