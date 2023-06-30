import os
import pickle
import codecs
import numpy as np

path = '/zcf/Flickr30k/data/f30k_bottom'


def read_pickle(file_path):
    with open(file_path, 'rb')as f:
        files = pickle.load(f)
    # print('read: ' + file_path)
    return files


def write_pickle(file_path, file):
    with open(file_path, 'wb')as f:
        pickle.dump(file, f)
    print('write: ' + file_path)


def read_txt(file_path):
    with codecs.open(file_path, 'r', 'utf-8')as f:
        list = f.readlines()
    print('read: ' + file_path)
    return list


def write_txt(file_path, file):
    with codecs.open(file_path, 'w', 'utf-8')as f:
        for line in file:
            f.write(str(line))
            f.write('\n')
    print('write: ' + file_path)


def write_txt_add(file_path, file):
    with codecs.open(file_path, 'a', 'utf-8')as f:
        for line in file:
            f.write(str(line))
            f.write('\n')
    print('write: ' + file_path)


def read_npy(file_path):
    file = np.load(file_path)
    # print('read: ' + file_path)
    return file


def write_npy(file_path, file):
    np.save(file_path, file)
    print('write: ' + file_path)


def part_tags():
    file_path = os.path.join(path, 'train_tags.pickle')
    tag = read_pickle(file_path)
    tag_part = tag[:29700]
    file_part_path = os.path.join(path, 'train_dnc_tags.pickle')
    write_pickle(file_part_path, tag_part)
    print('tags')


def part_sentence():
    file_path = os.path.join(path, 'f30k_train_sentence_list.txt')
    sentence = read_txt(file_path)
    sentence_part = sentence[:29700]
    file_part_path = os.path.join(path, 'train_dnc_sentence_list.txt')
    write_txt(file_part_path, sentence_part)
    print('sentence')


def part_image_list():
    file_path = os.path.join(path, 'train_image_list.txt')
    image_list = read_txt(file_path)
    image_list_part = image_list[:29700]
    file_part_path = os.path.join(path, 'train_dnc_image_list.txt')
    write_txt(file_part_path, image_list_part)
    print('image_list')


def part_feature():
    file_path = os.path.join(path, 'train_features.npy')
    feature = read_npy(file_path)
    feature_part = feature[:29700]
    feature_part_path = os.path.join(path, 'train_dnc_features.npy')
    write_npy(feature_part_path, feature_part)
    print('feature')


def part_boxes():
    file_path = os.path.join(path, 'train_boxes.npy')
    boxes = read_npy(file_path)
    boxes_part = boxes[:29700]
    boxes_part_path = os.path.join(path, 'train_dnc_boxes.npy')
    write_npy(boxes_part_path, boxes_part)
    print('boxes')


if __name__ == '__main__':
    print('start')
    part_tags()
    part_sentence()
    part_image_list()
    part_feature()
    part_boxes()
    print('end')
