import numpy as np
import copy
import torch
import math
import os
from torch import nn
import torch.nn.functional as F
import torch.fft
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder,AttackRTransformerEncoder
from recbole.model.loss import BPRLoss
from sklearn.cluster import KMeans, MiniBatchKMeans


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs 
    
class SGP(SequentialRecommender):
    r"""
    """

    def __init__(self, config, dataset, co_data, co_lens):
        super(SGP, self).__init__(config, dataset)
        
        self.co_seq = F.normalize(self.get_co(co_data,co_lens), dim=1).to(self.device)
        self.pos_emb = torch.nn.Embedding(self.max_seq_length, 300) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=0.5)
        self.last_layernorm = torch.nn.LayerNorm(300, eps=1e-8)
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.kk = 50
        self.kmeans = MiniBatchKMeans(n_clusters=self.kk, init_size=1024, batch_size=1024, random_state=100)
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.item_embedding = torch.nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        
        self.mb = 0.1
        self.ssl = 'us_x'
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.sem_aug_nce_fct = nn.CrossEntropyLoss()
        self.LayerNorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(0.5)
        self.mask_loss_weight = nn.Parameter(torch.FloatTensor([0.3]), requires_grad=True)
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(2):
            new_attn_layernorm = torch.nn.LayerNorm(300, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  torch.nn.MultiheadAttention(300,4,0.5)                                                               
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(300, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(300, 0.5)
            self.forward_layers.append(new_fwd_layer) 

        self.batch_size = 512
        self.d = 3
        self.knn_k = 10
        self.bal = 5
        self.miu_c = 0.3
        self.miu_m = 2.5
        self.knn_k_co = self.knn_k // self.bal
        self.knn_k_ma = self.knn_k - self.knn_k_co
        self.image_embedding = (dataset.image_embedding).to(self.device)
        self.text_embedding = (dataset.text_embedding).to(self.device)
        self.img_embs = (self.image_embedding.weight).to(self.device)
        self.text_embs = (self.text_embedding.weight).to(self.device)        
        self.co_img_embs = self.co_seq @ self.img_embs 
        self.co_text_embs = self.co_seq @ self.text_embs 
        indices, self.co_vm_adj = self.get_knn_adj_mat(self.img_embs)
        indices, self.co_tm_adj = self.get_knn_adj_mat(self.text_embs)
        self.sensev, self.vsample = self.get_center(self.img_embs)
        self.senset, self.tsample = self.get_center(self.text_embs)
        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def mod(self, build_item_graph=True):
        h = self.item_embedding.weight.to(self.device)
        vcoh = torch.mm(self.co_vm_adj, h)
        tcoh = torch.mm(self.co_tm_adj, h)     
        return h,vcoh,tcoh
    
    def forward(self, item_seq, item_seq_len):
        log_seqs = item_seq.cpu().numpy()
        ID,hmv_emb,hmt_emb = self.mod()  
        
        hv_after = torch.mm(self.sensev,self.co_vm_adj.to_dense())
        ht_after = torch.mm(self.senset,self.co_tm_adj.to_dense())
        hv_after = hv_after @ ID
        ht_after = ht_after @ ID     
        
        seqsv = hmv_emb[torch.LongTensor(log_seqs).to(self.device)]
        seqst = hmt_emb[torch.LongTensor(log_seqs).to(self.device)]
        seqsi = ID[torch.LongTensor(log_seqs).to(self.device)]
        cc = torch.tile(torch.arange(hv_after.shape[0]), (log_seqs.shape[0], 1))
        vsg=hv_after[cc]
        tsg=ht_after[cc]
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.device)

        # idco
        co_sensei = self.co_seq @ ID
        co_sense = co_sensei[torch.LongTensor(log_seqs).to(self.device)] 
        
        # id
        seqsi = co_sense 
        seqsi *= self.item_embedding.embedding_dim ** 0.9
        seqsi += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqsi = self.emb_dropout(seqsi)
        tl = seqsi.shape[1]
        attention_maski = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))
        seqsi *= ~timeline_mask.unsqueeze(-1) 

        # image
        seqsv *= self.item_embedding.embedding_dim ** 0.5
        seqsv += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqsv = self.emb_dropout(seqsv)
        tl = seqsv.shape[1]
        attention_maskv = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))
        
        seqsv *= ~timeline_mask.unsqueeze(-1)

        # text       
        seqst *= self.item_embedding.embedding_dim ** 0.5
        seqst += self.pos_emb(torch.LongTensor(positions).to(self.device))        
        seqst = self.emb_dropout(seqst)
        t2 = seqst.shape[1]
        attention_maskt = ~torch.tril(torch.ones((t2, t2), dtype=torch.bool, device=self.device))
        seqst *= ~timeline_mask.unsqueeze(-1)

        # star
        vsg = vsg * self.item_embedding.embedding_dim ** 0.5
        tsg = tsg * self.item_embedding.embedding_dim ** 0.5
        vsg = self.emb_dropout(vsg)
        tsg = self.emb_dropout(tsg)     
        suov = self.compute_max_similarity_index(seqsv,vsg)
        suot = self.compute_max_similarity_index(seqst,tsg)
        attsg = ~torch.tril(torch.ones((vsg.shape[1], t2), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):    
            vsg = torch.transpose(vsg, 0, 1)
            Qvsg = self.attention_layernorms[i](vsg)          
            seqsv = torch.transpose(seqsv, 0, 1)           
            Qv = self.attention_layernorms[i](seqsv)
            vvv, _= self.attention_layers[i](Qvsg, seqsv, seqsv, attn_mask=attsg)
            mha_outputsv, _= self.attention_layers[i](Qv, seqsv, seqsv, attn_mask=attention_maskv)     
            seqsv = Qv + mha_outputsv
            seqsv = torch.transpose(seqsv, 0, 1)
            seqsv = self.forward_layernorms[i](seqsv)
            seqsv = self.forward_layers[i](seqsv)
            seqsv *=  ~timeline_mask.unsqueeze(-1)                
            vsg = Qvsg + vvv
            vsg = torch.transpose(vsg, 0, 1)
            vsg = self.forward_layers[i](vsg)
              
            tsg = torch.transpose(tsg, 0, 1)
            Qtsg = self.attention_layernorms[i](tsg)
            seqst = torch.transpose(seqst, 0, 1)
            Qt = self.attention_layernorms[i](seqst)
            ttt, _= self.attention_layers[i](Qtsg, seqst, seqst, attn_mask=attsg)
            mha_outputst, _= self.attention_layers[i](Qt, seqst, seqst, attn_mask=attention_maskt) 
            seqst = Qt + mha_outputst
            seqst = torch.transpose(seqst, 0, 1)
            seqst = self.forward_layernorms[i](seqst)
            seqst = self.forward_layers[i](seqst)
            seqst *=  ~timeline_mask.unsqueeze(-1)
            tsg = Qtsg + ttt
            tsg = torch.transpose(tsg, 0, 1)
            tsg = self.forward_layers[i](tsg)

            seqsi = torch.transpose(seqsi, 0, 1)
            Qi = self.attention_layernorms[i](seqsi)
            mha_outputsi, _= self.attention_layers[i](Qi, seqsi, seqsi, attn_mask=attention_maski)
            seqsi = Qi + mha_outputsi
            seqsi = torch.transpose(seqsi, 0, 1)
            seqsi = self.forward_layernorms[i](seqsi)
            seqsi = self.forward_layers[i](seqsi)
            seqsi *=  ~timeline_mask.unsqueeze(-1)  

        cenv = torch.matmul(suov, vsg)
        cent = torch.matmul(suot, tsg)
        log_featsv = self.last_layernorm(seqsv + cenv)
        log_featst = self.last_layernorm(seqst + cent)
        outputv = self.gather_indexes(log_featsv, item_seq_len - 1) 
        outputt = self.gather_indexes(log_featst, item_seq_len - 1) 
        seqs = ((1 - self.mb) * (seqst) + self.mb * (seqsv) + 0.6 * (seqsi))

        log_feats = self.last_layernorm(seqs) 
        output = self.gather_indexes(log_feats, item_seq_len - 1)   
        return output, outputv, outputt  # [B H]    
    
    def compute_max_similarity_index(self, j, i):
        similarity = torch.matmul(j, i.transpose(1, 2))
        tensor_reshaped = similarity.view(-1, self.kk)  
        result_reshaped = F.gumbel_softmax(tensor_reshaped, tau=1, hard=False)
        result = result_reshaped.view(j.shape[0], j.shape[1], self.kk)
        return result

    def calculate_loss(self, interaction):
        _,hmv_emb,hmt_emb = self.mod()
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, outputv, outputt= self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            logits = torch.matmul(outputv, hmv_emb.transpose(0, 1))
            loss += self.mb * self.loss_fct(logits, pos_items)
            logits = torch.matmul(outputt, hmt_emb.transpose(0, 1))
            loss += (1 - self.mb) * self.loss_fct(logits, pos_items)

        return loss

    def full_sort_predict(self, interaction):
        h,vcoh,tcoh = self.mod()
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, outputv, outputt= self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        scores += self.mb * torch.matmul(outputv, vcoh.transpose(0, 1))  # [B]
        scores += (1 - self.mb) * torch.matmul(outputt, tcoh.transpose(0, 1))  # [B] 
        return scores 
    
    def get_co(self, seqs,len):
        co_mat = torch.zeros(self.n_items,self.n_items)
        for i in range(seqs.shape[0]):
            for k in range(len[i]):
                for j in range(k+1,len[i]):
                    co_mat[seqs[i][k]][seqs[i][j]] +=1/(j-k) 
                    co_mat[seqs[i][j]][seqs[i][k]] +=1/(j-k) 
        return co_mat

    def extract_common_and_complement(self, a, b, n):
        m, _ = a.shape
        c = torch.full((m, n), -1, dtype=torch.int64)

        for i in range(m):
            row_a = a[i].tolist()
            row_b = b[i].tolist()
            common_elements = list(set(row_a) & set(row_b))
            remaining_elements = [x for x in row_a if x not in common_elements]
            c[i][:len(common_elements)] = torch.tensor(common_elements, dtype=torch.int64)
            c[i][len(common_elements):] = torch.tensor(remaining_elements[:n - len(common_elements)], dtype=torch.int64)
        
        return c.cuda()

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind_ma = torch.topk(sim, int(self.knn_k * self.miu_m), dim=-1)
        _, knn_ind_co = torch.topk(self.co_seq, int(self.knn_k * self.miu_c), dim=-1)
        knn_ind = self.extract_common_and_complement(knn_ind_ma, knn_ind_co, self.knn_k)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)
        
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size) 
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_center(self,embs):
        means = embs.detach().cpu().numpy()
        self.kmeans.fit(means)
        centers = torch.tensor(self.kmeans.cluster_centers_).to(self.device)
        sample = torch.tensor(self.kmeans.labels_)
        o = torch.zeros(1, self.n_items).to(self.device)
        for i in range(max(sample)+1):
            op=copy.deepcopy(sample).unsqueeze(0).to(self.device)
            for j in range(self.n_items-1):
                if op[0,j]==i:
                    op[0,j]=1
                else:
                    op[0,j]=0
            o=torch.cat((o, op), 0)
        sense=o[1:]
        sense = sense/(1e-7+torch.sum(sense,dim=-1).unsqueeze(1))
        return sense,sample
