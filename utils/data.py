import h5py
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from itertools import groupby


def encode_texts(texts, blank_idx=0):
    def _label_to_num(label):
        label_num = []
        for ch in label:
            idx = alphabet.find(ch)
            label_num.append(idx + (idx >= blank_idx))
        return np.array(label_num)

    alphabet = ''.join(sorted(set(''.join(texts))))
    if blank_idx < 0:
        blank_idx = len(alphabet)
    nums = np.full([len(texts), max([len(text) for text in texts])], fill_value=blank_idx, dtype='int64')
    for i, text in enumerate(texts):
        nums[i][:len(text)] = _label_to_num(text)

    return nums, alphabet


def decode_texts(logits, alphabet, blank_idx):
    if blank_idx < 0:
        blank_idx = len(alphabet)
    best_path_indices = np.argmax(logits, axis=-1)
    best_chars_collapsed = [[alphabet[idx-(idx >= blank_idx)] for idx, _ in groupby(e) if idx != blank_idx]
                            for e in best_path_indices]
    return [''.join(e) for e in best_chars_collapsed]


def load_data(data_path='./data', seed=42, split=True, blank_idx=0):
    with h5py.File(os.path.join(data_path, 'common_fields_images.h5')) as f:
        images = f['images'][:]
        additional_bits = f['additional_bit'][:]

    with open(os.path.join(data_path, 'common_fields_labels.txt'), encoding='cp1251') as f:
        markup = [e.strip() for e in f.readlines()]

    images = images.astype('float32') / 255

    additional_bits_expanded = np.zeros((len(images), 50, 2))
    additional_bits_expanded[:, :, additional_bits] = 1

    labels_encoded, alphabet = encode_texts(markup, blank_idx=blank_idx)

    if split:
        np.random.seed(seed)

        train_indices = np.random.choice(np.arange(images.shape[0]), int(images.shape[0] * 0.8), replace=False)
        val_indices = [e for e in np.arange(images.shape[0]) if e not in train_indices]

        assert len(set(train_indices) & set(val_indices)) == 0
        assert len(set(train_indices) | set(val_indices)) == images.shape[0]

        train_imgs = images[train_indices]
        val_imgs = images[val_indices]

        train_abits = additional_bits_expanded[train_indices]
        val_abits = additional_bits_expanded[val_indices]

        train_labels = labels_encoded[train_indices]
        val_labels = labels_encoded[val_indices]

        return ((train_imgs, train_abits), train_labels), ((val_imgs, val_abits), val_labels), alphabet

    else:
        return ((images, additional_bits_expanded), labels_encoded), alphabet


class OCRDataset(Dataset):
    def __init__(self, images, abits, labels):
        super(OCRDataset, self).__init__()

        self.images = images
        self.abits = abits
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.images[idx]).unsqueeze(0), torch.FloatTensor(self.abits[idx])), torch.IntTensor(self.labels[idx])
