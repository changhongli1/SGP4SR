import os.path as osp
import math
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset
import copy
import torch.nn.functional as F

class IndexableBuffer:
    def __init__(self, data):
        self.data = data
        self.num_embeddings = len(data)
        self.embedding_dim = data.shape[1] if data.ndim > 1 else None

    def __getitem__(self, index):
        return self.data[index]

    @property
    def device(self):
        return self.data.device
    
    def __setitem__(self, index, val):
        self.data[index] = val

    def __call__(self, index=None):
        return self.__getitem__(index)


class SGPDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
        self.text_size = config['text_size']
        self.image_size = config['image_size'] if hasattr(config, 'image_size') else config['text_size']
        self.text_suffix = config['text_suffix']
        self.image_suffix = config['image_suffix']
        text_embedding_weight = self.load_text_embedding()
        self.text_embedding = self.weight2emb(text_embedding_weight, self.text_size)
        image_embedding_weight = self.load_image_embedding()
        self.image_embedding = self.weight2emb(image_embedding_weight, self.image_size)
        
    def init_mapper(self):
        self.iid2id = {}
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]':
                continue
            self.iid2id[int(token)] = i

        self.uid2id = {}
        for i, token in enumerate(self.field2id_token['user_id']):
            if token == '[PAD]':
                continue
        self.uid2id[int(token)] = i
    
    def load_text_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.text_suffix}')
        loaded_feat = np.load(feat_path, allow_pickle=True)
        mapped_feat = np.zeros((self.item_num, self.text_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]':
                continue
            mapped_feat[i] = loaded_feat[int(token)]
        return mapped_feat

    def load_image_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.image_suffix}')
        loaded_feat = np.load(feat_path, allow_pickle=True)
        mapped_feat = np.zeros((self.item_num, self.image_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]':
                continue
            mapped_feat[i] = loaded_feat[int(token)]
        return mapped_feat

    def weight2emb(self, weight, emd_size):
        text_embedding = nn.Embedding(self.item_num, emd_size, padding_idx=0)
        text_embedding.weight.requires_grad = False
        text_embedding.weight.data.copy_(torch.from_numpy(weight))
        return text_embedding
    
    def prepare_data_augmentation(self):
        self.logger.debug('prepare_data_augmentation')
        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        uu = torch.tensor(self.inter_feat[self.uid_field])
        for i, uid in enumerate(uu.numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)
        self.uid_list = np.array(uid_list)
        self.item_list_index = np.array(item_list_index)
        self.target_index = np.array(target_index)
        self.item_list_length = np.array(item_list_length, dtype=np.int64)
        self.mask = np.ones(len(self.inter_feat), dtype=np.bool_)

    def extend_repeats_with_condition(self, tensor):
        result = []
        prev_value = None
        repeat_count = 0
        for i, value in enumerate(tensor):
            if value == prev_value:
                repeat_count += 1
            else:
                if repeat_count >= 2:
                    result.extend([prev_value] * (repeat_count + 1))
                elif repeat_count == 1: 
                    result.append(prev_value)
                repeat_count = 1
                prev_value = value

            if i == len(tensor) - 1:
                if repeat_count >= 2:
                    result.extend([value] * (repeat_count + 1))
                else:  
                    result.append(value)

        return torch.tensor(result, dtype=tensor.dtype)
    
    def semantic_augmentation(self, target_index):
        
        same_target_index = []
        tt = torch.tensor(self.inter_feat['item_id'])
        target_item = (tt[target_index].numpy())
        for index, item_id in enumerate(target_item):
            all_index_same_id = np.where(target_item == item_id)[0]  
            delete_index = np.argwhere(all_index_same_id == index)
            all_index_same_id_wo_self = np.delete(all_index_same_id, delete_index)
            same_target_index.append(all_index_same_id_wo_self)
        same_target_index = np.array(same_target_index)
        return same_target_index

    def leave_one_out(self, group_by, leave_one_num=1):
        self.logger.debug(f'Leave one out, group_by=[{group_by}], leave_one_num=[{leave_one_num}].')
        if group_by is None:
            raise ValueError('Leave one out strategy require a group field.')
        if group_by != self.uid_field:
            raise ValueError('Sequential models require group by user.')

        self.prepare_data_augmentation()
        grouped_index = self._grouped_index(self.uid_list)
        next_index = self._split_index_by_leave_one_out(grouped_index, leave_one_num)
        self._drop_unused_col()
        next_ds = []
        for index in next_index:
            ds = copy.copy(self)
            # print(ds)
            for field in ['uid_list', 'item_list_index', 'target_index', 'item_list_length']:
                setattr(ds, field, np.array(getattr(ds, field)[index]))
            setattr(ds, 'mask', np.ones(len(self.inter_feat), dtype=np.bool_))
            
            next_ds.append(ds)
        next_ds[0].mask[self.target_index[next_index[1] + next_index[2]]] = False
        next_ds[1].mask[self.target_index[next_index[2]]] = False
        self.same_target_index = self.semantic_augmentation(next_ds[0].target_index)
        setattr(next_ds[0], 'same_target_index', self.same_target_index)
        return next_ds
