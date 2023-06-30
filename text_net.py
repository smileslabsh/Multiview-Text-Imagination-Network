import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Parameter
import torchvision as tv

import tokenization
from bert import BertConfig, BertModel
import bert
from torch.autograd import Variable
import cupy
from data import convert_to_feature
from sentence_transformers import SentenceTransformer


def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


class BertMapping(nn.Module):
    """
    """
    def __init__(self, opt):
        super(BertMapping, self).__init__()
        bert_config = BertConfig.from_json_file(opt.bert_config_file)
        self.bert = BertModel(bert_config)
        self.bert.load_state_dict(torch.load(opt.init_checkpoint, map_location='cpu'))
        freeze_layers(self.bert)
        self.txt_stru = opt.txt_stru
        self.encode_model = SentenceTransformer('bert-base-uncased')

        self.attn_matrix = nn.Linear(768, opt.final_dims, bias=False)

        if opt.txt_stru == 'pooling':
            self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
            self.mapping = nn.Linear(bert_config.hidden_size, opt.final_dims)
        elif opt.txt_stru == 'cnn':
            Ks = [1, 2, 3]
            in_channel = 1
            out_channel = 512
            embedding_dim = bert_config.hidden_size
            self.convs1 = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, embedding_dim)) for K in Ks])
            self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
            self.mapping = nn.Linear(len(Ks)*out_channel+256, opt.final_dims)
        elif opt.txt_stru == 'rnn':
            embedding_dim = bert_config.hidden_size
            self.bi_gru = opt.bi_gru
            self.rnn = nn.GRU(embedding_dim, opt.embed_size, opt.num_layers, batch_first=True, bidirectional=opt.bi_gru)
            self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
            self.mapping = nn.Linear(opt.embed_size, opt.final_dims)
        elif opt.txt_stru == 'trans':
            bert_config = BertConfig.from_json_file(opt.img_trans_cfg)
            self.layer = bert.BERTLayer(bert_config)
            self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
            self.mapping = nn.Linear(768, opt.final_dims)

        self.init_weights()

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.attn_matrix.in_features +
                                  self.attn_matrix.out_features)
        self.attn_matrix.weight.data.uniform_(-r, r)

    def forward(self, input_ids, attention_mask, token_type_ids, lengths, img_code, expand_words):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        if self.txt_stru == 'pooling':
            output = self.mapping(all_encoder_layers[-1])
            output = torch.mean(output, 1)
        elif self.txt_stru == 'cnn':
            x = all_encoder_layers[-1].unsqueeze(1)  # (batch_size, 1, token_num, embedding_dim)
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(batch_size, out_channel, W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
            output = torch.cat(x, 1)
        elif self.txt_stru == 'rnn':
            x = all_encoder_layers[-1]  # (batch_size, token_num, embedding_dim)
            packed = pack_padded_sequence(x, lengths, batch_first=True)
            # Forward propagate RNN
            out, _ = self.rnn(packed)
            # Reshape *final* output to (batch_size, hidden_size)
            padded = pad_packed_sequence(out, batch_first=True)
            cap_emb, cap_len = padded
            if self.bi_gru:
                cap_emb = (cap_emb[:, :, :cap_emb.size(2) / 2] + cap_emb[:, :, cap_emb.size(2) / 2:]) / 2
            else:
                cap_emb = cap_emb
            output = torch.mean(cap_emb, 1)
        elif self.txt_stru == 'trans':
            hidden_states = self.mapping(all_encoder_layers[-1])
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.float()
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            hidden_states = self.layer(hidden_states, extended_attention_mask)
            # output = hidden_states[:, 0, :]
            output = torch.mean(hidden_states, 1)

        img_code = img_code.cuda()
        expand_feature = self.get_expand_feature(img_code, expand_words)  # [batch_size, 256]
        output = torch.cat((output, expand_feature), dim=1)
        output = self.dropout(output)  # [batch_size, 1536]
        code = self.mapping(output)
        # code = F.tanh(code)
        code = F.normalize(code, p=2, dim=1)
        return code

    def get_expand_feature(self, img_code, expand_words):
        size = img_code.size()
        attn_embed = torch.zeros(size[0], 256)
        num_words = 3

        for i in range(size[0]):
            image_fea = img_code[i]

            # expand_word_fea = [cupy.asnumpy(self.bert_spacy(j.split('_')[0]).vector) for j in expand_words[i]]  # [3, 768]
            expand_word_fea = []  # [3, 256]
            expand_word_w = []  # [3]

            for j in range(min(len(expand_words[i]), num_words)):
                l = expand_words[i][j].split('_')
                word, weight = l[0], l[1]

                if word == 'None':
                    expand_word_fea.append(torch.zeros(768))
                    expand_word_w.append(0.3)
                else:
                    expand_word_fea.append(torch.Tensor(self.encode_model.encode(word, show_progress_bar=False)))
                    expand_word_w.append(float(weight))


            # val = torch.tensor([item.cpu().detach().numpy() for item in expand_word_fea]).cuda()
            expand_word_fea = torch.tensor([item.cpu().detach().numpy() for item in expand_word_fea]).cuda()
            expand_word_w = torch.Tensor(expand_word_w).cuda()

            expand_word_fea = self.attn_matrix(expand_word_fea)
            # [3, 256]
            img_temp = image_fea.unsqueeze(0)
            # [1, 256]
            attn_mat = torch.mm(img_temp, expand_word_fea.transpose(0, 1)).squeeze()
            # [1, 256] * [256, 3] -> [1, 3] -> 3

            # img_temp = self.attn_matrix(image_fea)
            # # 256 -> 768
            # img_temp = img_temp.unsqueeze(0)
            # # [1, 768]
            # attn_mat = torch.mm(img_temp, expand_word_fea.transpose(0, 1)).squeeze()
            # # [1, 768] * [768, 3] -> [1, 3] -> 3

            attn_mat = nn.Tanh()(attn_mat)
            attn_mat = nn.Softmax()(attn_mat)
            attn_mat = attn_mat * expand_word_w

            norm_mat = torch.sum(attn_mat).repeat(3)
            attn_mat = attn_mat / norm_mat
            attn_mat = torch.mm(attn_mat.unsqueeze(0), expand_word_fea).squeeze()

            attn_embed[i] = attn_mat.data


        attn_embed = Variable(attn_embed).cuda()
        # [60, 256]
        return attn_embed
