import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


class BiEncoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs['bert']

    def forward(self, context_input_ids, context_input_masks,
                            responses_input_ids, responses_input_masks, labels=None):
        ## only select the first response (whose lbl==1)
        if labels is not None:
            responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)

        context_vec = self.bert(context_input_ids, context_input_masks)[0][:,0,:]  # [bs,dim]

        batch_size, res_cnt, seq_length = responses_input_ids.shape
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)

        responses_vec = self.bert(responses_input_ids, responses_input_masks)[0][:,0,:]  # [bs,dim]
        responses_vec = responses_vec.view(batch_size, res_cnt, -1)

        if labels is not None:
            responses_vec = responses_vec.squeeze(1)
            dot_product = torch.matmul(context_vec, responses_vec.t())  # [bs, bs]
            mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device)
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            context_vec = context_vec.unsqueeze(1)
            dot_product = torch.matmul(context_vec, responses_vec.permute(0, 2, 1)).squeeze()
            return dot_product


class CrossEncoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs['bert']
        self.linear = nn.Linear(config.hidden_size, 1)

    def forward(self, text_input_ids, text_input_masks, text_input_segments, labels=None):
        batch_size, neg, dim = text_input_ids.shape
        text_input_ids = text_input_ids.reshape(-1, dim)
        text_input_masks = text_input_masks.reshape(-1, dim)
        text_input_segments = text_input_segments.reshape(-1, dim)
        text_vec = self.bert(text_input_ids, text_input_masks, text_input_segments)[0][:,0,:]  # [bs,dim]
        score = self.linear(text_vec)
        score = score.view(-1, neg)
        if labels is not None:
            loss = -F.log_softmax(score, -1)[:,0].mean()
            return loss
        else:
            return score


class PolyEncoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs['bert']
        self.poly_m = kwargs['poly_m']
        self.poly_code_embeddings = nn.Embedding(self.poly_m, config.hidden_size)
        # https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/transformer/polyencoder.py#L355
        torch.nn.init.normal_(self.poly_code_embeddings.weight, config.hidden_size ** -0.5)

    def dot_attention(self, q, k, v):
        # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        # k=v: [bs, length, dim] or [bs, poly_m, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1))  # [bs, poly_m, length]
        # print('attn_weights.shape 1',attn_weights.shape)
        attn_weights = F.softmax(attn_weights, -1)
        # print('attn_weights.shape 2', attn_weights.shape)
        output = torch.matmul(attn_weights, v)  # [bs, poly_m, dim]
        # print('output.shape', output.shape)
        return output

    def forward(self, context_input_ids, context_input_masks,
                responses_input_ids, responses_input_masks, labels=None):
        # during training, only select the first response
        # we are using other instances in a batch as negative examples
        if labels is not None:
            responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)
        batch_size, res_cnt, seq_length = responses_input_ids.shape  # res_cnt is 1 during training

        # context encoder
        # context_out is a mtrix with dim of [batch_size, length, embedding_size]
        ctx_out = self.bert(context_input_ids, context_input_masks)[0]  # [bs, length, dim]
        # print('ctx_out.shape',ctx_out.shape)

        # poly_code_ids=[0,1,2...,poly_m]
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(context_input_ids.device)
        # print('poly_code_ids',poly_code_ids)

        # give each sample in a batch a [0,1,2...,poly_m], so the dim is [batch_size, poly_m].
        # there poly_code_ids is the Query in  the Figure 1. So the Query is position id.
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        # print('poly_code_ids.shape',poly_code_ids.shape)
        # print('poly_code_ids',poly_code_ids)

        # turn each position id to an embedding
        poly_codes = self.poly_code_embeddings(poly_code_ids)  # [bs, poly_m, dim]
        # print('poly_codes.shape',poly_codes.shape)

        # return Ctxt Emb
        embs = self.dot_attention(poly_codes, ctx_out, ctx_out)  # [bs, poly_m, dim]

        # response encoder
        # print('responses_input_ids.shape bef', responses_input_ids.shape)
        # from [bs, 1, 32] to [bs, 32]. 32 is the max_reponse_len
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        # print('responses_input_ids.shape aft', responses_input_ids.shape)
        responses_input_masks = responses_input_masks.view(-1, seq_length)

        # print('self.bert(responses_input_ids, responses_input_masks)[0].shape', \
        #       self.bert(responses_input_ids, responses_input_masks)[0].shape)  # [bs,32,128]
        # only pick up the first element in the dim of length！
        cand_emb = self.bert(responses_input_ids, responses_input_masks)[0][:, 0, :]  # [bs, dim]
        # print('cand_emb.shape',cand_emb.shape)
        # res_cnt is 1 during training. so it will be [32, 1, 128]
        cand_emb = cand_emb.view(batch_size, res_cnt, -1)  # [bs, res_cnt=1, dim]
        # print('cand_emb.shape',cand_emb.shape)

        # merge
        if labels is not None:
            # we are recycling responses for faster training
            # we repeat responses for batch_size times to simulate test phase
            # so that every context is paired with batch_size responses
            cand_emb = cand_emb.permute(1, 0, 2)  # [1, bs, dim]
            cand_emb = cand_emb.expand(batch_size, batch_size, cand_emb.shape[2])  # [bs, bs, dim]
            # print('cand_emb.shape', cand_emb.shape)
            # print('cand_emb.shape',cand_emb[0])
            # print('cand_emb.shape',cand_emb[1])

            # print(self.dot_attention(cand_emb, embs, embs).shape)
            # print(self.dot_attention(cand_emb, embs, embs).squeeze().shape)
            # there is no dim number 1, so squeeze() dose not influence
            # print('embs.shape', embs.shape)
            # self.dot_attention([bs, bs, dim],[bs, poly_m, dim])
            ctx_emb = self.dot_attention(cand_emb, embs, embs).squeeze()  # [bs, bs, dim]
            # print('ctx_emb.shape', ctx_emb.shape)
            # print('ctx_emb.shape',ctx_emb[0])
            # print('ctx_emb.shape',ctx_emb[1])
            # what's the meaning of ctx_emb? It is for each sample (bs=64 samples)
            # 他的第一个维度来自于batch（64个sample）,第二个维度来自于query也就是cand emd就是64，第三个维度来自
            # value也就是embs,它的意思应该是64个sample下的64的response的检索结果(context 结果)
            # 每个sample的区别是他们的context不同，所以我们可以认为是64个context下。

            # cand emb的意思是有64次的以下数据：64个response下的64个response（16个eponse list）的embedding
            # 他们相乘，就会获得64个context_dot_response下的，64个 response的检索结果_dot_response, 的context结果_dot_response_embedding
            # 这下子在三个维度上都交互了
            # print('ctx_emb*cand_emb.shape', (ctx_emb * cand_emb).shape)
            dot_product = (ctx_emb * cand_emb).sum(-1)  # [bs, bs]
            # embedding加总，得到的就是64个context_dot_response下的，64个 response的检索结果_dot_response的值
            # print('dot_product.shape', dot_product.shape)
            mask = torch.eye(batch_size).to(context_input_ids.device)  # [bs, bs]
            # print('mask.shape', mask.shape)
            # 这里让loss变小，就是让对角线变大，就是让context_dot_response下的对应的response的检索结果_dot_response的值最大。
            # 至于那些不对应的值不管它们！！
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            ctx_emb = self.dot_attention(cand_emb, embs, embs)  # [bs, res_cnt, dim]
            dot_product = (ctx_emb * cand_emb).sum(-1)
            return dot_product
