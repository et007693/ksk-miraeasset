import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

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
        outputs = outputs.transpose(-1, -2) 
        outputs += inputs
        return outputs

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_ratio, device):
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim # 임베딩 차원
        self.num_heads = num_heads # 헤드(head)의 개수: 서로 다른 어텐션(attention) 컨셉의 수
        self.head_dim = hidden_dim // num_heads # 각 헤드(head)에서의 임베딩 차원

        self.fc_q = nn.Linear(hidden_dim, hidden_dim) # Query 값에 적용될 FC 레이어
        self.fc_k = nn.Linear(hidden_dim, hidden_dim) # Key 값에 적용될 FC 레이어
        self.fc_v = nn.Linear(hidden_dim, hidden_dim) # Value 값에 적용될 FC 레이어

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # hidden_dim → num_heads X head_dim 형태로 변형
        # num_heads(h)개의 서로 다른 어텐션(attention) 컨셉을 학습하도록 유도
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention Energy 계산
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 마스크(mask)를 사용하는 경우
        if mask is not None:
            # 마스크(mask) 값이 0인 부분을 -1e10으로 채우기
            energy = energy.masked_fill(mask, 0)

        # 어텐션(attention) 스코어 계산 - 각 단어에 대한 확률 값
        attention = torch.softmax(energy, dim=-1)

        # Scaled Dot-Product Attention 계산
        x = torch.matmul(self.dropout(attention), V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hidden_dim)

        return x, attention

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, item_side, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.item_side = item_side
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.side_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) 
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms1 = torch.nn.ModuleList()
        self.attention_layernorms_side = torch.nn.ModuleList()
        self.attention_layers1 = torch.nn.ModuleList()
        self.forward_layernorms1 = torch.nn.ModuleList()
        self.forward_layers1 = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) 
            self.attention_layernorms1.append(new_attn_layernorm)
            self.attention_layernorms_side.append(new_attn_layernorm)

            new_attn_layer = MultiHeadAttentionLayer(args.hidden_units, args.num_heads, args.dropout_rate,self.dev)
            self.attention_layers1.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms1.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers1.append(new_fwd_layer)


    def log2feats(self, log_seqs, side_seqs):
        # 스케일링
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        seqs_s1 = self.side_emb(torch.LongTensor(side_seqs).to(self.dev))
        seqs_s1 *= self.side_emb.embedding_dim ** 0.5
        
        # emb + position
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        seqs1 = seqs + seqs_s1
        seqs1 += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs1 = self.emb_dropout(seqs1)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)

        seqs1 *= ~timeline_mask.unsqueeze(-1) 

        tl = seqs1.shape[1] 
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        for i in range(len(self.attention_layers1)):
            
            Q1 = self.attention_layernorms1[i](seqs1)
            mha_outputs1, _ = self.attention_layers1[i](Q1, seqs1, seqs1, mask=attention_mask)
            seqs1 = Q1 + mha_outputs1
            seqs1 = self.forward_layernorms1[i](seqs1)
            seqs1 = self.forward_layers1[i](seqs1)
            seqs1 *=  ~timeline_mask.unsqueeze(-1)
        
        log_feats = self.last_layernorm(seqs1) 

        return log_feats


    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, log_seqs_side):  
        log_feats = self.log2feats(log_seqs, log_seqs_side)

        pos_embs = (self.item_emb(torch.LongTensor(pos_seqs).to(self.dev)) )
        neg_embs = (self.item_emb(torch.LongTensor(neg_seqs).to(self.dev)) )

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits 

    def predict(self, user_ids, log_seqs, item_indices, log_seqs_side, item_indices_side): 
        
        log_feats = self.log2feats(log_seqs, log_seqs_side)

        final_feat = log_feats[:, -1, :] 
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) 
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
