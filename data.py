import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import tokenization
import parts
from torch.utils.data import Sampler
import random
import pickle


def convert_to_feature(raw, seq_length, tokenizer):
    line = tokenization.convert_to_unicode(raw)
    tokens_a = tokenizer.tokenize(line)
    # Modifies `tokens_a` in place so that the total
    # length is less than the specified length.
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return tokens, input_ids, input_mask, input_type_ids


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, opt):
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc+'%s_sentence_list_final.txt' % data_split) as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        # self.images = np.load(loc+'%s_ims.npy' % data_split)
        def sort_image(elem):
            elem_num = elem.split('.')
            return int(elem_num[0])

        self.images_path = loc + data_split + '_features'
        self.images = os.listdir(self.images_path)
        self.images.sort(key=sort_image)

        self.images_tag_path = loc + data_split + '_tags_vector_new'
        self.images_tag = os.listdir(self.images_tag_path)
        self.images_tag.sort(key=sort_image)
        
        self.length = len(self.captions)

        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=opt.vocab_file, do_lower_case=opt.do_lower_case)
        self.max_words = opt.max_words

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index / self.im_div)
        # image = torch.Tensor(self.images[img_id])
        image = torch.Tensor(parts.read_npy(os.path.join(self.images_path, self.images[img_id])))
        image_tag = torch.Tensor(parts.read_npy(os.path.join(self.images_tag_path, self.images_tag[img_id])))
        caption = self.captions[index]
        # caption = "Five people wearing winter clothing , helmets , and ski goggles stand outside in the snow .#person_0.1,white snow_0.2,snow man_0.4"
        # split = caption.decode().rsplit('.', 1)
        split = caption.rsplit('#', 1)
        caption, expand_words = split[0], split[1].split(',')[:3]

        tokens, input_ids, input_mask, input_type_ids = convert_to_feature(caption, self.max_words, self.tokenizer)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        input_type_ids = torch.tensor(input_type_ids, dtype=torch.long)
        return image, image_tag, input_ids, input_mask, input_type_ids, index, img_id, expand_words

    def __len__(self):
        return self.length


class MySampler(Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        num = len(self.data_source)
        self.final_index = []

        with open('./st_clustered_id_50.pickle', 'rb') as f:
            clustered_id = pickle.load(f)

        cluster_index = [[] for i in range(len(clustered_id))]
        for i in range(len(clustered_id)):
            for j in clustered_id[i]:
                temp = [5 * j + k for k in torch.randperm(5).tolist()]
                cluster_index[i].append(temp)

        for i in range(5):
            for j in range(len(cluster_index)):
                randperj = torch.randperm(len(cluster_index[j])).tolist()
                for k in randperj:
                    self.final_index.append(cluster_index[j][k][i])

    def __iter__(self):
        return iter(self.final_index)

    def __len__(self):
        return len(self.final_index)


def collate_fn_bert(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: sum(x[3]), reverse=True)
    images, images_tag, input_ids, input_mask, input_type_ids, ids, img_ids, expand_words = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    images_tag = torch.stack(images_tag, 0)
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [torch.sum(cap) for cap in input_mask]
    input_ids = torch.stack(input_ids, 0)
    input_mask = torch.stack(input_mask, 0)
    input_type_ids = torch.stack(input_type_ids, 0)

    ids = np.array(ids)

    return images, images_tag, input_ids, input_mask, input_type_ids, lengths, ids, expand_words


def get_precomp_loader(data_path, data_split, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, opt)

    if data_split == 'train':
        # mySampler = MySampler(dset, batch_size)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  # sampler=mySampler,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn_bert)
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn_bert)
    return data_loader


def get_loaders(data_name, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, opt,
                                     batch_size, False, workers)
    return test_loader
