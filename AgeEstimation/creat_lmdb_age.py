import os
import glob
import random
import numpy as np
import cv2
import sys
sys.path.append("/home/genesis/.local/install/caffe/python/")
import caffe
from caffe.proto import caffe_pb2
import lmdb

# Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    # image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())


train_lmdb = './train_lmdb'
validation_lmdb = './validation_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)
index = 6
lastndx = 0
print 'train_lmdb olusturuluyor...'

train_data = [img for img in glob.glob("./dataset/*jpg")]

age0 = 0
age20 = 0
age40 = 0
age60 = 0
random.shuffle(train_data)
in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):

        if in_idx % 6 == 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        for age in img_path[10:len(img_path)].split("_"):
            # print age
            break
        if int(age) < 20:
            label = 0
            age0 += 1
        elif int(age) < 40:
            label = 1
            age20 += 1
        elif int(age) < 60:
            label = 2
            age40 += 1
        else:
            label = 3
            age60 += 1
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path + ':' + str(label)
in_db.close()

print '\nvalidation_lmdb olusturuluyor'

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        if in_idx % 6 != 0:
            continue
        for age in img_path[10:len(img_path)].split("_"):
            # print age
            break
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if int(age) < 20:
            label = 0
            age0 += 1
        elif int(age) < 40:
            label = 1
            age20 += 1
        elif int(age) < 60:
            label = 2
            age40 += 1
        else:
            label = 3
            age60 += 1
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path + ':' + str(label)
in_db.close()
print '\nFinished processing all images'
print "0-20:" + str(age0)
print "20-40:" + str(age20)
print "40-60:" + str(age40)
print "60+:" + str(age60)
