# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# from natsort import natsorted, ns
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import numpy as np
import torch
import os
from itertools import combinations
from . import data_utils, FairseqDataset
from tqdm import tqdm
from tqdm import tqdm_notebook
logger = logging.getLogger(__name__)
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import gc
import multiprocessing
import torch.multiprocessing as mp

def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, src_img_features=None
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def multigraph_merge(txt_key, multimodel_key, src_lengths):
        return data_utils.collate_multimodel_graphs(
            [s[txt_key] for s in samples],
            [s[multimodel_key] for s in samples],src_lengths
        )

    def merge1(key):
        return data_utils.collate_tokens1(
            [s[key] for s in samples]
        )



    def generate_pairs(root_folder):
        for folders in root_folder:
            if len(folders) == 1:

                single_feature = np.load(folders[0], allow_pickle=True)['arr_0']
                noise = np.random.normal(0, 0.1, single_feature.shape)
                yield (folders[0], single_feature + noise)
            else:
                for pair in combinations(folders[:2], 2):
                    yield pair


    def generate_pairs_EMMT(root_folder):
        for folders in root_folder:

            noise = np.random.normal(0, 0.1, folders.shape)
            yield (folders, folders + noise)

    def generate_pairs_Multi30k(root_folder):
        for folders in root_folder:

            noise = np.random.normal(0, 0.1, folders.shape)
            yield (folders, folders + noise)



    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()


    def process_features(src_img_features):
        with ThreadPoolExecutor(max_workers=64) as executor:

            path_to_future = {fp: executor.submit(load_feature, fp) for fp in
                              set(fp for pair in src_img_features for fp in pair if isinstance(fp, str))}

            all_features = []
            for pair in src_img_features:

                feature_1 = pair[0] if isinstance(pair[0], np.ndarray) else path_to_future[pair[0]].result()
                feature_2 = pair[1] if isinstance(pair[1], np.ndarray) else path_to_future[pair[1]].result()
                all_features.append([feature_1, feature_2])
        return all_features

    def load_feature(file_path):

        loaded_data = np.load(file_path, allow_pickle=True)
        return loaded_data['arr_0'] if 'arr_0' in loaded_data else file_path


    id = torch.LongTensor([s['id'] for s in samples])

    src_tokens = merge('source', left_pad=left_pad_source)

    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)

    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    #######################Fashion-MMT#######################
    # if samples[0].get('src_img_features') is not None:

        # src_img_features = list(generate_pairs([s['src_img_features'] for s in samples]))

        # all_features = process_features(src_img_features)
    #########################################################

    ########################EMMT##############################
    # if samples[0].get('src_img_features') is not None:
    #     src_img_features = list(generate_pairs_EMMT([s['src_img_features'] for s in samples]))

    #     all_features = process_features(src_img_features)
    ##########################################################

    ########################Multi-30k##############################
    if samples[0].get('src_img_features') is not None:
        src_img_features = list(generate_pairs_Multi30k([s['src_img_features'] for s in samples]))

        all_features = process_features(src_img_features)
    ##########################################################

    #     ########################WIT##############################
    # if samples[0].get('src_img_features') is not None:
    #     src_img_features = list(generate_pairs_Multi30k([s['src_img_features'] for s in samples]))
    #
    #     all_features = process_features(src_img_features)
    #     ##########################################################


        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'src_img_features': all_features,
                # 'src_img_features': src_img_features,
                # 'src_img_features_location': src_img_features
                # 'multimodel_graph':multimodel_graph
            },
            'target': target,
        }
    else:
        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,
        }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        src_img_features=None,
        # bpe_txt_relations=None,
        # img_txt_relations=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        append_bos=False
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src_img_features = src_img_features
        # self.bpe_txt_relations = bpe_txt_relations
        # self.img_txt_relations = img_txt_relations
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos
        # self.preloaded_file_paths = self._preload_file_paths()
        # self.preloaded_file_paths_1 = self._preload_file_paths_1()
        # self.preloaded_file_paths = self.get_first_n_file_paths(110257)


    def get_first_n_file_paths(self, n):
        base_folder = '/Fashion-MMT'
        # Create file paths for the first n numeric file names
        file_paths = [os.path.join(base_folder, str(i)) for i in range(1, n + 1)]
        return file_paths



    def __getitem__(self, index, src_img_features=None):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        if self.src_img_features is not None:
        ############################Fashion-MMT#############################
            # src_img_features = [os.path.join(self.preloaded_file_paths[index-1], i) for i in os.listdir(self.preloaded_file_paths[index-1])]
        ####################################################################
            src_img_features = self.src_img_features[index].reshape(1,49,2048)
        ############################EMMT####################################

            # src_img_features = self.src_img_features[index]

        ###############################Multi30k###################################
            # src_img_features = self.src_img_features[index]
        ##########################################################################

        ###############################WIT###################################
        # src_img_features = self.src_img_features[index]
        ##########################################################################

        else:
            src_img_features = None
        # src_img_features_item = img_tensor.view(img_tensor.size(0), img_tensor.size(1) * img_tensor.size(2), -1)
        # bpe_txt_relations_item = self.bpe_txt_relations[index]
        # img_txt_relations_item = self.img_txt_relations[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
        if src_img_features is not None:
            example = {
                'id': index,
                'source': src_item,
                'target': tgt_item,
                # 'src_img_features': src_img_features.unsqueeze(0),
                'src_img_features':src_img_features
                # 'bpe_txt_relations':bpe_txt_relations_item,
                # 'img_txt_relations':img_txt_relations_item,
            }
        else:
            example = {
                'id': index,
                'source': src_item,
                'target': tgt_item
            }
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)


