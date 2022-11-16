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
        '''
        *args称之为Non-keyword Variable Arguments，无关键字参数；
        **kwargs称之为keyword Variable Arguments，有关键字参数；
        当函数中以列表或者元组的形式传参时，就要使用*args；
        当传入字典形式的参数时，就要使用**kwargs。
        https://zhuanlan.zhihu.com/p/144773033
        '''
        super().__init__(config, *inputs, **kwargs) #TODO 有空了查查这个super为什么要这么用
        self.bert = kwargs['bert']
        self.poly_m = kwargs['poly_m']
        self.poly_code_embeddings = nn.Embedding(self.poly_m, config.hidden_size)
        # https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/transformer/polyencoder.py#L355
        # 这里是对embedding进行初始化
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
        print('responses_input_ids.shape bef:', responses_input_ids.shape)
        if labels is not None:
            # 一上来不知道这里是什么意思，但是可以根据这里进行大概的推断。首先这里有三个维度，第一个一般是bs，第二个一般是就需要具体看了。
            # 参考下面一行可以知道，第二个维度是res_cnt，第三个维度是seq_length
            # 这里其实就是选取了第一个response，也就是正样本那个
            responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)
        batch_size, res_cnt, seq_length = responses_input_ids.shape  # res_cnt is 1 during training
        print('responses_input_ids.shape aft:', responses_input_ids.shape)
        ''' 注意,这里的context_input_ids是[bs, 16, max_contexts_length],而
        responses_input_ids是[bs,max_response_length]. 这个16是怎么来的那?是在parse.py就写死了的,就是每个context只选择包含1个positive的
        response和15个negative的response. 
        这里处理之后responses_input_ids就变成了[bs, 1, max_contexts_length]
        '''

        # context encoder
        # context_out is a mtrix with dim of [bs, length, embedding_size]
        # print('context_input_ids.shape:', context_input_ids.shape)
        # print('context_input_ids:', context_input_ids)
        ctx_out = self.bert(context_input_ids, context_input_masks)[0]  # [bs, length, dim] #这种写法可读性不够好
        # print('ctx_out.shape',ctx_out.shape)


        ctx_out_check = ctx_out.sum(-1)
        print('ctx_out_check.shape', ctx_out_check.shape)
        print('ctx_out_check', ctx_out_check)


        # poly_code_ids=[0,1,2...,poly_m] poly_m默认设置是16，就是生成一个[1,2,3,...16]的向量
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(context_input_ids.device)
        # print('poly_code_ids',poly_code_ids)

        # give each sample in a batch a [0,1,2...,poly_m], so the dim is [bs, poly_m].
        # there poly_code_ids is the Query in  the Figure 1. So the Query is position id.
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        # print('poly_code_ids.shape',poly_code_ids.shape)
        # print('poly_code_ids',poly_code_ids.shape)

        # turn each position id to an embedding
        poly_codes = self.poly_code_embeddings(poly_code_ids)  # [bs, poly_m, dim]
        # print('poly_codes.shape',poly_codes.shape)


        poly_codes_check = poly_codes.sum(-1)
        print('poly_codes_check.shape', poly_codes_check.shape)
        print('poly_codes_check', poly_codes_check)

        # return Ctxt Emb
        '''对于dot_attention返回的维度，第一个维度是bs这个不变，第二个维度来自于Q的第二个维度，第三个维度来自V的维度。
        以如下的Q=poly_codes=[bs, poly_m, dim], K=V=ctx_out=[bs, ctx_length, dim]为例，
        首先是用Q和K算出来一个attention_weight=[bs,poly_m,ctx_length]代表着每一个Q对于每一个K的查询结果；
        其次用attention_weight和Vmutual，算出来的是weight之后的Value=[bs,poly_m,dim]代表着每一个Query的查询结果
        '''
        embs = self.dot_attention(poly_codes, ctx_out, ctx_out)  # [bs, poly_m, dim]

        embs_check = embs.sum(-1)
        print('embs_check.shape', embs_check.shape)
        print('embs_check', embs_check)

        attn_weights_check = torch.matmul(poly_codes, ctx_out.transpose(2, 1))  # [bs, poly_m, length]
        print('attn_weights_check.shape', embs_check.shape)
        print('attn_weights_check', embs_check)

        # response encoder
        # from [bs, 1, 32] to [bs, 32]. 32 is the max_reponse_len
        # print('responses_input_ids.shape bef', responses_input_ids.shape)
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        # print('responses_input_ids.shape aft', responses_input_ids.shape)
        # print('responses_input_masks.shape bef', responses_input_masks.shape)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        # print('responses_input_masks.shape aft', responses_input_masks.shape)

        # print('self.bert(responses_input_ids, responses_input_masks)[0].shape', \
        #       self.bert(responses_input_ids, responses_input_masks)[0].shape)  # [bs,32,128]
        # only pick up the first element in the dim of length！ 以起始位置来代表整个的embedding
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
            cand_emb = cand_emb.permute(1, 0, 2)  # 交换维度，从[bs, res_cnt=1, dim]到[1, bs, dim]

            # expand,目标维度是[bs, bs, dim].这种维度的扩张一般是为了加下来进行矩阵运算
            cand_emb = cand_emb.expand(batch_size, batch_size, cand_emb.shape[2])
            # print('cand_emb.shape', cand_emb.shape)
            # print('cand_emb.shape',cand_emb[0])
            # print('cand_emb.shape',cand_emb[1])

            # print(self.dot_attention(cand_emb, embs, embs).shape)
            # print(self.dot_attention(cand_emb, embs, embs).squeeze().shape)
            # there is no dim number 1, so squeeze() dose not influence
            # print('embs.shape', embs.shape)
            # self.dot_attention([bs, bs, dim],[bs, poly_m, dim])
            # 简而言之,这里进行了进一步的attention,得到一个新的attention之后的embedding
            '''Q=cand_emb: [bs, bs, dim]
               K=V=embs: [bs, poly_m, dim]
               ctx_emb: [bs, bs(from Q), dim (from V)]
               
               他的第一个维度来自于batch（一个batch有bs个sample）,第二个维度来自于query也就是cand emd就是bs，第三个维度来自
            value也就是embs,它的意思应该是bs个sample下的bs的response的检索结果(context值) 
            '''
            ctx_emb = self.dot_attention(cand_emb, embs, embs).squeeze()  # [bs, bs, dim]
            # print('ctx_emb.shape', ctx_emb.shape)
            # print('ctx_emb.shape',ctx_emb[0])
            # print('ctx_emb.shape',ctx_emb[1])
            # what's the meaning of ctx_emb? It is for each sample (bs=64 samples)


            # cand_emb_check = cand_emb.sum(axis=-1)
            # print('cand_emb_check', cand_emb_check)
            # print('cand_emb_check.shape', cand_emb_check.shape)

            ''' 这个模型的原始输入是bs个context和bs个response(而且这个response是positive的response).
                对于ctx emb,他是[bs, bs, dim],它是怎么来的那? 
                ctx_emb = self.dot_attention(cand_emb, embs, embs).squeeze()  # [bs, bs, dim]
                要先看下cand_emb.
                cand_emb 本来是一个[bs, dim]的,但是把它重复了bs次,变成了[bs,bs,dim].相当于本来是
                [vec_1, vec_2, vec_3,..., vec_bs]，现在变成了
                [[vec_1, vec_2, vec_3,..., vec_bs],
                 [vec_1, vec_2, vec_3,..., vec_bs],
                 [vec_1, vec_2, vec_3,..., vec_bs],
                 ...
                 [vec_1, vec_2, vec_3,..., vec_bs]]
                 在第一个维度上重复了 bs次
                (参见 cand_emb_check)
                当cand_emb是[bs,dim]的时候,它代表着bs个response的emb;
                
                现在看下embs怎么来.
                embs = self.dot_attention(poly_codes, ctx_out, ctx_out)  # [bs, poly_m, dim]
                其中ctx_out是[bs, length, dim], 是bs个context下每一个token的embedding 
                poly_codes来自于 poly_codes = self.poly_code_embeddings(poly_code_ids)  # [bs, poly_m, dim]
                poly_codes是重复了bs次的数字的embs,即[vec_0, vec_1,...,vec_poly_m]重复了bs次
                embs所以是ctx的emb,是bs个context经过了 poly_m次检索后得到的attn_context_emb 
                
                所以ctx_emb emb的意思是: bs个context下每个response_attention下的context_embed
                
                因cand_emb是: bs个response下的bs个response的response_emb,
                所以ctx_emb*cand_emb [bs, bs, dim] 就如我之前在邮件里写的那样,第一个bs代表着bs个pairwise对(context-respond对);
                以[1,bs,edim]为例,代表着第一个context所对应的bs个repsonse的response_attention下的context_embed和bs个response_embed
            '''
            # print('ctx_emb.shape', ctx_emb.shape)
            # print('cand_emb.shape', cand_emb.shape)
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
            # 如果没有label,那么就是在inference阶段...待
            ctx_emb = self.dot_attention(cand_emb, embs, embs)  # [bs, res_cnt, dim]
            dot_product = (ctx_emb * cand_emb).sum(-1)
            return dot_product
