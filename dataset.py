import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import random
import pickle


#observe what the SelectionDataset dose
from torch.utils.data import Dataset
class SelectionDataset(Dataset):
    def __init__(self, file_path, context_transform, response_transform, concat_transform, sample_cnt=None, mode='poly'):
        self.context_transform = context_transform
        self.response_transform = response_transform
        self.concat_transform = concat_transform
        self.data_source = []
        self.mode = mode
        neg_responses = []
        with open(file_path, encoding='utf-8') as f:
            group = {
                'context': None,
                'responses': [],
                'labels': []
            }

            # i = 0
            for line in f:

                # # 1 observe lines
                # i += 1
                # if i <=100:
                #   print('line:',line)

                split = line.strip('\n').split('\t')
                # label, cntext(就是之前产生的所有对话. participant 1: balabala articipant 2:balabala),
                # response 就是一段文本

                lbl, context, response = int(split[0]), split[1:-1], split[-1]
                # 这里的if是一个判断停止条件，真正的主体句子在if-else后面
                if lbl == 1 and len(group['responses']) > 0:
                    # print("group['responses']", group['responses'])
                    # 想要理解上面if的用处，就必须要从if里面的内容来看。if里的内容
                    # 这里内容的意思就是"添加并清空group对象"。结合上面的if条件，意思就是
                    # 如果遇到了label是1且group的response里非空，那么就添加并清空。
                    # 为什么这样写那？因为他的所有的数据都经过了处理，第一个都是label为1.
                    # 这样写就是为了当已经操作完第一条数据后，当遇到第二条数据的时候就是Label是1且有response，
                    # 那么就把第一条存储下，并且清空字典准备放第二条数据。
                    # 其实也可以把判断条件放在主体语句（三个group语句）的后面，然后用context是否发生变化来
                    # 判断，但是这样的话用context判断估计会对运行效率产生影响
                    self.data_source.append(group)
                    group = {
                        'context': None,
                        'responses': [],
                        'labels': []
                    }
                    # sample的数量上线。sample是指group这样的。
                    # if判断语句有前置的也有后置的，后置的比较容易。前置的那种必须要结合if条件和条件里的主体才行
                    if sample_cnt is not None and len(self.data_source) >= sample_cnt:
                        break
                else:
                        neg_responses.append(response) #这个neg_responses好像并没有使用
                # 这里是主体句子。加到group里
                group['responses'].append(response)
                group['labels'].append(lbl)
                group['context'] = context
            if len(group['responses']) > 0:
                self.data_source.append(group)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        group = self.data_source[index]
        context, responses, labels = group['context'], group['responses'], group['labels']
        # 2 observe the data beofre transform
        # print('--')
        # print('context', context)
        # print('responses', responses)
        # print('len(responses)', len(responses))
        # print('labels', labels)

        if self.mode == 'cross':
            transformed_text = self.concat_transform(context, responses)
            ret = transformed_text, labels
        else:
            transformed_context = self.context_transform(context)  # [token_ids],[seg_ids],[masks]
            transformed_responses = self.response_transform(responses)  # [token_ids],[seg_ids],[masks]
            ret = transformed_context, transformed_responses, labels

        # print('ret', ret)
        return ret #这里返回的是一个tuple。我觉得这个也可以，或者返回一个字典会更好。
        # 注意这里返回一个tuple貌似是因为__getitem__这种就是要返回一个list里面以tuple为元素

    # https://zhuanlan.zhihu.com/p/346332974

    def batchify_join_str(self, batch):
        # print('type(batch)', type(batch))
        # print('type(batch)', batch)

        if self.mode == 'cross':
            text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = [], [], []
            labels_batch = []
            for sample in batch:
                text_token_ids_list, text_input_masks_list, text_segment_ids_list = sample[0]

                text_token_ids_list_batch.append(text_token_ids_list)
                text_input_masks_list_batch.append(text_input_masks_list)
                text_segment_ids_list_batch.append(text_segment_ids_list)

                labels_batch.append(sample[1])

            long_tensors = [text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch]

            text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = (
                torch.tensor(t, dtype=torch.long) for t in long_tensors)

            labels_batch = torch.tensor(labels_batch, dtype=torch.long)
            return text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch, labels_batch

        else:
            #!https://androidkt.com/create-dataloader-with-collate_fn-for-variable-length-input-in-pytorch/#:~:text=A%20custom%20collate_fn%20can%20be,from%20the%20data%20loader%20iterator.
            # https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders
            # https://zhuanlan.zhihu.com/p/346332974
            contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
            responses_token_ids_list_batch, responses_input_masks_list_batch = [], [], [], []
            labels_batch = []
            for sample in batch: # 这里batch的输入是__getitem__所返回的一个list[tuple]的对象
                # 对于每一个 sample，因为它是一个tuple=(transformed_context, transformed_responses, labels)
                # 这里类似于一个解包操作，他和前面__getitem__里的是对应的
                (contexts_token_ids_list, contexts_input_masks_list), \
                (responses_token_ids_list, responses_input_masks_list) = sample[:2] #是右闭区间

                contexts_token_ids_list_batch.append(contexts_token_ids_list)
                contexts_input_masks_list_batch.append(contexts_input_masks_list)

                responses_token_ids_list_batch.append(responses_token_ids_list)
                responses_input_masks_list_batch.append(responses_input_masks_list)

                labels_batch.append(sample[-1])

            long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch,
                                            responses_token_ids_list_batch, responses_input_masks_list_batch]

            contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
            responses_token_ids_list_batch, responses_input_masks_list_batch = (
                torch.tensor(t, dtype=torch.long) for t in long_tensors)
            # 这里的操作相当于是把ret=(transformed_context, transformed_responses, labels)
            # 给拆解出来，变成了return里面的格式，且都是torch.long格式的
            labels_batch = torch.tensor(labels_batch, dtype=torch.long)
            return contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
                          responses_token_ids_list_batch, responses_input_masks_list_batch, labels_batch
