'''
1. 关于DST 7 https://ibm.github.io/dstc-noesis/public/data_description.html
这类描述的是比较准确的
2. 在这之前，已经经过了parse.sh处理了
3. 对比ubuntu_train_subtask_1.json的第一个example和train.txt的第一个example，可以知道在train里，每一行是label(0\1)\t'context'\t'response'. 可以知道的是context产生了大量的重复。
'''
import argparse
import sys

from torch.utils.data import DataLoader

from dataset import SelectionDataset
import os

from transform import SelectionJoinTransform, SelectionSequentialTransform, SelectionConcatTransform
from transformers import BertModel, BertConfig, BertTokenizer, BertTokenizerFast

import logging
logging.basicConfig(level=logging.ERROR)

sys.argv = ['-f']

parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--bert_model", default='ckpt/pretrained/bert-small-uncased', type=str)
parser.add_argument("--eval", action="store_true")
parser.add_argument("--model_type", default='bert', type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--train_dir", default='data/ubuntu_data', type=str)

parser.add_argument("--use_pretrain", action="store_true")
parser.add_argument("--architecture", type=str, help='[poly, bi, cross]')

parser.add_argument("--max_contexts_length", default=128, type=int)
parser.add_argument("--max_response_length", default=32, type=int)
parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
parser.add_argument("--print_freq", default=100, type=int, help="Log frequency")

parser.add_argument("--poly_m", default=0, type=int, help="Number of m of polyencoder")

parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--warmup_steps", default=100, type=float)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

parser.add_argument("--num_train_epochs", default=10.0, type=float,
                                        help="Total number of training epochs to perform.")
parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                                        help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
              "See details at https://nvidia.github.io/apex/amp.html",
)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
print(args)

#手动设置参数
parser.set_defaults(bert_model='bert_model')
parser.set_defaults(output_dir='output_dstc7')
parser.set_defaults(train_dir='dstc7')
parser.set_defaults(use_pretrain=True)
parser.set_defaults(architecture='poly')
parser.set_defaults(poly_m=16)
args = parser.parse_args()
print(args)

MODEL_CLASSES = {
        'bert': (BertConfig, BertTokenizerFast, BertModel),
    }
ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]
train_dir = 'dstc7'
tokenizer = TokenizerClass.from_pretrained(os.path.join(args.bert_model, "vocab.txt"), do_lower_case=True, clean_text=False)
context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=args.max_contexts_length)
response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=args.max_response_length)
concat_transform = SelectionConcatTransform(tokenizer=tokenizer, max_len=args.max_response_length+args.max_contexts_length)

train_dataset = SelectionDataset(os.path.join(train_dir, 'train.txt'),
                                                                  context_transform, response_transform, concat_transform, sample_cnt=None, mode=args.architecture)

# observe the train_dataset
transformed_context, transformed_responses, labels = train_dataset.__getitem__(0)
# print('transformed_context',transformed_context)

# 观察context_transform
context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=args.max_contexts_length)

context=['participant 1: Hi, I want to run a graphical application from the command line, here is the script I wrote: https://paste8.com/4XQiHrXZ - it\'s Ubuntu Server 12.04 + Unity. What I get is an error from xhost "unable to open display :0" and from the graphical application I want to use (Sikuli) "Can\'t connect to X11 window server using \':0\' as the value of the DISPLAY variable.". I\'ve tried using DISPLAY:=1 as I use this number when connecting with a VNC client but it doesn\'t wo. rk either...', 'participant 2: is X running?', 'participant 1: I think so: https://paste8.com/HvhlT6vO.  maybe you prefer this check: https://paste8.com/0OhcBmfB', 'participant 2: these days you use lightdm (or gdm, kdm) to start X, so i\'d try to kill what\'s there but unreachable with "sudo killall X" and then "sudo service lightdm start"', 'participant 1: what do I kill precisely? because it says "X: no such process"', 'participant 2: nothing, X wasn\'t running if you get that, proceed with "sudo service lightdm start" and then test your script', 'participant 1: alright but "start: Job is already running: lightdm"', 'participant 2: worm: try restart', 'participant 1: I did but I get "No protocol specified. " followed by "xhost: unable to open display :0" when executing my script (i.e. https://paste8.com/4XQiHrXZ). does it matter I run everything from a root tty? because that\'s what I do', 'participant 2: ahh, not a good idea to try to start X as root no.  now your regular users ~/.Xauthority file is probably owned by root, check that.  what does "ls -la ~/.Xauthority" give you?', 'participant 1: yes I got /root/.Xauthority I need to append an entry for my regular user right?', 'participant 2: check if your regular user still owns his .Xauthority file and remove the one for root', "participant 1: there is no .Xauthority for my regular user I'm afraid", 'participant 2: well, copy the one from root to the user /home and make him own it, use the chown command.  sudo chown $USER:$USER $HOME/.Xauthority', 'participant 1: done, but running $sudo service lightdm start; returns "Sorry, user <myuser> is not allowed to execute \'/usr/sbin/service lightdm start\' as root on <myhost>."', 'participant 2: is the regular user in the sudoers?', "participant 1: no, I believe I have to create a file in /etc/sudoers.d but I don't know the syntax yet"]
transformed_context = context_transform(context)

# 这个故意创造了一个比较短的句子，于是就有了padding在右侧
context=['participant 1: Hi, I want to ', "participant 2: no, I believe I have to create"]
transformed_context = context_transform(context)

# 观察 response_transform
response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=args.max_response_length)
responses=["are you still running in root terminal? if so, try from a non-root one.. i've got to go prepare dinner so i'll be afk .. ask the channel for help on that, hope you get things up", 'quite', "hey join #.  so i can tell you things that the kiddies won't latch on to", 'did you complete the ubuntu installation??', 'how did you run mplayer?', "a skype replacement? i think google talk works in the US (i wouldn't know i'm in uk)", "you only need the libraries for each.... ... which isn't too expensive", 'try sudo apt-get install ubuntu-desktop', "which error is that?. btw don't make us ask these questions", "I'm not sure there is one, you may need to just tweak ~/.config/xfce4.", "- you'll figure it out - best to you.", 'try running "sudo updatedb" then "locate eclipse". youll find it then', "I don't remember though. anyone else? how to get verbose output during a boot? turn of boot splash during live cd?", "It's painful that's true :p.", 'Try cat /etc/issue.  Sorry, debian_version is for the debian version, not the ubuntu version', '1 mini pastbin']
transformed_responses = response_transform(responses)
# have a look at some element
# print(transformed_responses[0][0])
# print(transformed_responses[1][0])

# observe the train data
# 它的输出长度是5，输出的类型就从 collate_fn 里使用的 batchify_join_str 的return可以看出来
train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.batchify_join_str, shuffle=True, num_workers=0)
one_sample = next(iter(train_dataloader))

# print('type(one_sample)', type(one_sample))
# print(len(one_sample))
# print(len(one_sample[0]))  #0-4 is all 32 len
# print(one_sample[0])

