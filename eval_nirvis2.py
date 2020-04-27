#!/usr/bin/env python

"""Run face recognition on CASIA-NIR-VIS2 (cropped)


Usage: ./eval_nirvis2.py <model> <prototxt> <data_dir> <save_feat_path> [options]


Arguments:
  <model>           Path to the CNN model weights
  <prototxt>        The deploy file name
  <data_dir>        Path to the root of cropped casia_nirvis2 dataset which include a 'protocols' folder
  <save_feat_path>  Path to save features

Options:
  -h, --help              Shows this help message and exits
  --gpu                   Use GPU
  --gray                  Use gray images
  --num_classes=<int>     Number of classes used for training model [default: 10575]
  --debug

Examples:

     $ ./eval_nirvis2.py <path/to/model> <prototxt> <data_dir> <tmp/nirvis2_casianet.pkl>

     <path/to/model>=/path/to/rcn10_NIR_VIS.caffemodel
     <prototxt>=/path/to/deploy.prototxt
     <data_dir>=/path/to/CASIA_NIR-VIS2.0

"""

import docopt
import os, sys
import numpy as np
import time
import matplotlib.pyplot as plt
sys.path.insert(0, './utils/')  # /path/to/utils/
from casia_nirvis2 import NIRVIS2

import common
from sklearn.metrics.pairwise import cosine_similarity

caffe_root = '/path/to/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

def get_feat(img, count_num, modality, bnorm=True):
    global net

    net.blobs['data'].data[...] = img
    output = net.forward()

     if modality == 'vis':
         feat = net.blobs['fc5'].data[0:count_num]
         feat = np.hstack((feat, net.blobs['fc5'].data[count_num:2 * count_num]))
     else:
         feat = net.blobs['feat_probe'].data[0:count_num]
         feat = np.hstack((feat, net.blobs['feat_probe'].data[count_num:2 * count_num]))

    # normalize
    if bnorm:
        std = np.sqrt((feat ** 2).sum(-1))
        for i in range(feat.shape[0]):
            feat[i, :] /= (std[i] + 1e-10)

    return feat

global net
args = docopt.docopt(__doc__)
_DEBUG_ = args['--debug']

from scipy.misc import imread

if __name__ == '__main__':
    print args
    save_path = args['<save_feat_path>']
    if args['--gray']:
        image_loader = common.image_loader_gray
    else:
        image_loader = common.image_loader_rgb

    nirvis = NIRVIS2(args['<data_dir>'])
    nirvis.get_eval_flists()

    if args['--gpu']:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    net = caffe.Net(args['<prototxt>'], args['<model>'], caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    mean = np.array([127.5, 127.5, 127.5])
    transformer.set_mean('data', mean)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    # extract feature for probes and galleries of VIEW 2
    assert len(nirvis.eval_probe_gallery_flists) == 10
    acc = []
    VR_FAR_01 = []
    feats = {}
    if os.path.exists(save_path):
        common.print_info('read cmc scores from {}'.format(save_path))
        feats = common.read_pkl(save_path)

    # check whether image file exists
    if _DEBUG_:
        for probe_txt, gallery_txt in nirvis.eval_probe_gallery_flists:
            count_probe = 0
            count_gallery = 0
            with open(os.path.join(nirvis.path_protocol, probe_txt)) as fid:
                for line in fid:
                    image_path = os.path.join(nirvis.root, line.rstrip().replace('\\', '/'))
                    if os.path.exists(image_path):
                        count_probe += 1
                    else:
                        print image_path
            with open(os.path.join(nirvis.path_protocol, gallery_txt)) as fid:
                for line in fid:
                    image_path = os.path.join(nirvis.root, line.rstrip().replace('\\', '/'))
                    if os.path.exists(image_path):
                        count_gallery += 1
                    else:
                        print image_path
            print gallery_txt, count_gallery
            print probe_txt, count_probe

    split = 1
    for probe_txt, gallery_txt in nirvis.eval_probe_gallery_flists:
        this_fold_probe_feats = {}
        this_fold_gallery_feats = {}

        print probe_txt, gallery_txt
        start_time = time.time()
        batch_size = 256
        tmp_feat = []
        inputs = np.zeros([batch_size, 3, 112, 96], dtype=np.float32)

        transformed_image = np.zeros([3, 112, 96], dtype=np.float32)
        count_num = 0
        with open(os.path.join(nirvis.path_protocol, probe_txt)) as fid:
            for line in fid:
                image_path = os.path.join(nirvis.root, line.rstrip().replace('\\', '/'))

                imfile = imread(image_path)
                if imfile.shape.__len__() == 2:
                    imfile = imfile[np.newaxis, :, :]
                    imfile = np.concatenate((imfile, imfile, imfile), axis=0)
                else:
                    imfile = imfile.transpose((2, 0, 1))
                transformed_image[2, :, :] = imfile[0, :, :]
                transformed_image[1, :, :] = imfile[1, :, :]
                transformed_image[0, :, :] = imfile[2, :, :]
                transformed_image = (transformed_image - 127.5) / 128
                # flip
                flipped_img = transformed_image[:, ::-1, :]
                inputs[count_num, :, :, :] = transformed_image
                inputs[count_num + batch_size / 2, :, :, :] = flipped_img
                count_num += 1
                if count_num == batch_size / 2:
                    if len(tmp_feat) == 0:
                        tmp_feat = get_feat(inputs, count_num, 'nir')
                    else:
                        tmp_feat = np.vstack((tmp_feat, get_feat(inputs, count_num, 'nir')))
                    count_num = 0
        tmp = get_feat(inputs, batch_size / 2, 'nir')
        tmp_feat = np.vstack((tmp_feat, tmp[0:count_num, :]))

        with open(os.path.join(nirvis.path_protocol, probe_txt)) as fid:
            i = 0
            for line in fid:
                this_fold_probe_feats[line.rstrip()] = tmp_feat[i, :]
                feats[line.rstrip()] = this_fold_probe_feats[line.rstrip()]
                i += 1
        probe_feat = tmp_feat

        tmp_feat = []
        count_num = 0
        with open(os.path.join(nirvis.path_protocol, gallery_txt)) as fid:
            for line in fid:
                image_path = os.path.join(nirvis.root, line.rstrip().replace('\\', '/'))
                imfile = imread(image_path)
                if imfile.shape.__len__() == 2:
                    imfile = imfile[np.newaxis, :, :]
                    imfile = np.concatenate((imfile, imfile, imfile), axis=0)
                else:
                    imfile = imfile.transpose((2, 0, 1))

                transformed_image[2, :, :] = imfile[0, :, :]
                transformed_image[1, :, :] = imfile[1, :, :]
                transformed_image[0, :, :] = imfile[2, :, :]
                transformed_image = (transformed_image - 127.5) / 128

                # flip
                flipped_img = transformed_image[:, ::-1, :]

                inputs[count_num, :, :, :] = transformed_image
                inputs[count_num + batch_size / 2, :, :, :] = flipped_img
                count_num += 1
                if count_num == batch_size / 2:
                    if len(tmp_feat) == 0:
                        tmp_feat = get_feat(inputs, count_num, 'vis')
                    else:
                        tmp_feat = np.vstack((tmp_feat, get_feat(inputs, count_num, 'vis')))
                    count_num = 0

        tmp = get_feat(inputs, batch_size / 2, 'vis')
        tmp_feat = np.vstack((tmp_feat, tmp[0:count_num, :]))

        with open(os.path.join(nirvis.path_protocol, gallery_txt)) as fid:
            i = 0
            for line in fid:
                this_fold_gallery_feats[line.rstrip()] = tmp_feat[i, :]
                feats[line.rstrip()] = this_fold_gallery_feats[line.rstrip()]
                i += 1
        gallery_feat = tmp_feat

        if _DEBUG_:
            print('Elapsed time(s): %f' % (time.time() - start_time))
        # calcu mask matries
        cs = cosine_similarity(np.array(this_fold_probe_feats.values()), np.array(this_fold_gallery_feats.values()))

        # ======== rank1 performance with mask
        cmc_scores = []
        roc_positive = []
        roc_negative = []
        for i, k in enumerate(this_fold_probe_feats.keys()):
            probe_id = k.split('\\')[2]
            positives = []  # for genuine accesses
            negatives = []  # for impostor accesses
            for ii, kk in enumerate(this_fold_gallery_feats.keys()):
                gallery_id = kk.split('\\')[2]
                if probe_id == gallery_id:
                    positives.append(cs[i, ii])
                    roc_positive.append(cs[i, ii])
                else:
                    negatives.append(cs[i, ii])
                    roc_negative.append(cs[i, ii])
            cmc_scores.append((np.array(negatives), np.array(positives)))

        acc_rank1 = common.recognition_rate(cmc_scores, None, rank=1)

        common.print_info('{}'.format(acc_rank1))
        acc.append(acc_rank1)

        fpr, tpr, thresholds, extra = common.roc(np.array(roc_negative), np.array(roc_positive), True)
        VR_FAR_01.append(extra['ACC@FAR_0.1%'])

    common.data_to_pkl(feats, save_path)
    # average performance on 10 folds
    common.print_info(
        'CASIA-NIR-VIS2, Rank-1 Accuracy, {} std, {}'.format(np.mean(np.array(acc)), np.std(np.array(acc))))
    common.print_info(
        'CASIA-NIR-VIS2, VR@FAR=0.1%, {} std, {}'.format(np.mean(np.array(VR_FAR_01)), np.std(np.array(VR_FAR_01))))
