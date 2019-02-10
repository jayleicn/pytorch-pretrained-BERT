from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Jie Lei"
"""
Extract BERT contextualized token embedding, using [1]. Modified from [4].


Instructions:
0, This code should be running at Python 3.5+ and PyTorch 0.4.1/1.0
1, Input is a jsonl file. Each line is a json object, containing a 
   {"id": *str_id*, "text": *space separated sequence*}. Output is 
   also a jsonl file, each line contains 
2, Tokens that are split into subword by WordPiece, their embeddings are 
   the averaged embedding of its subword embeddings, as in the [2, 3]. This
   makes the output embedding respect the original tokenization scheme.
3, Each of the subtitle, question, and answers are encoded separately, 
   not in the form of sequence pairs. This means they are all `sequence A`, 
   `A embedding` is added to each of the token embeddings before forward 
   into the model. See [0] for details.
 

[0] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
[1] https://github.com/huggingface/pytorch-pretrained-BERT
[2] From Recognition to Cognition: Visual Commonsense Reasoning
[3] SDNET: CONTEXTUALIZED ATTENTION-BASED DEEP NETWORK FOR CONVERSATIONAL QUESTION ANSWERING
[4] https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/extract_features.py
"""


import argparse
import collections
import logging
import json
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
import base64
import h5py

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def pad_sequences_1d(sequences, dtype=torch.long):
    """ Pad a single-nested list or a sequence of n-d torch tensor into a (n+1)-d tensor,
        only allow the first dim has variable lengths
    Args:
        sequences: list(n-d tensor or list)
        dtype: torch.long for word indices / torch.float (float32) for other cases
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
    """
    if isinstance(sequences[0], list):
        sequences = [torch.tensor(s, dtype=dtype) for s in sequences]
    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros((len(sequences), max(lengths)) + extra_dims, dtype=dtype)
    mask = torch.zeros(len(sequences), max(lengths)).float()
    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


def pad_collate(data):
    batch = edict()
    batch["token_ids"], batch["token_ids_mask"] = pad_sequences_1d([d.token_ids for d in data], dtype=torch.long)
    batch["token_map"] = [d.token_map for d in data]
    batch["unique_id"] = [d.unique_id for d in data]
    batch["tokens"] = [d.tokens for d in data]
    batch["original_tokens"] = [d.original_tokens for d in data]
    return batch


class BertSingleSeqDataset(Dataset):
    def __init__(self, input_file, bert_model, max_seq_len):
        self.examples = self.read_examples(input_file)
        print("There are {} lines".format(len(self.examples)))
        self.bert_full_tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.wordpiece_tokenizer = self.bert_full_tokenizer.wordpiece_tokenizer
        self.convert_tokens_to_ids = self.bert_full_tokenizer.convert_tokens_to_ids
        self.convert_ids_to_tokens = self.bert_full_tokenizer.convert_ids_to_tokens
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        text = example["text"].lower()
        original_tokens = text.split()
        wp_tokens, wp_token_ids, wp_map = self.normalize_text(original_tokens, self.max_seq_len)

        items = edict(
            unique_id=example["unique_id"],
            tokens=wp_tokens,
            token_ids=wp_token_ids,
            token_map=wp_map,
            original_tokens=original_tokens,
        )
        return items

    @staticmethod
    def read_examples(input_file):
        """Read a list of `InputExample`s from an input jsonl file,
        {"id": *str_id*, "text": *space separated sequence, tokenized, lowercased*}"""
        examples = []
        with open(input_file, "r", encoding='utf-8') as reader:
            while True:
                line = reader.readline()
                if not line:
                    break
                line = json.loads(line.strip())
                examples.append(
                    edict(unique_id=line["id"], text=line["text"]))
        return examples

    def wordpiece_tokenize(self, tokens):
        """tokens (list of str): tokens from another tokenizer"""
        wp_tokens = []
        wp_token_map = []
        for t in tokens:
            wp_token_map.append(len(wp_tokens))
            wp_tokens.extend(self.wordpiece_tokenizer.tokenize(t))
        return wp_tokens, wp_token_map

    def normalize_text(self, tokens, max_seq_length):
        """convert to a word piece tokenized list of tokens
        from pre-tokenized tokens"""
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        wp_tokens, wp_token_map = self.wordpiece_tokenize(tokens)
        if len(wp_tokens) > max_seq_length - 2:
            wp_tokens = wp_tokens[0:(max_seq_length - 2)]
            max_idx_location = np.searchsorted(wp_token_map, max_seq_length - 2, side="right")
            wp_token_map = wp_token_map[:max_idx_location]
        wp_tokens = ["[CLS]"] + wp_tokens + ["[SEP]"]
        wp_token_map = np.array(wp_token_map) + 1  # account for "[CLS]" token
        wp_token_map = wp_token_map.tolist()
        wp_token_map.append(-1)  # the last token is "[SEP]", which we will remove
        wp_token_ids = self.convert_tokens_to_ids(wp_tokens)
        return wp_tokens, wp_token_ids, wp_token_map

    def compare_tokenization(self, text):
        assert text.islower()
        original_tokens = text.split()

        wp_tokens, wp_token_ids, wp_map = self.normalize_text(original_tokens, self.max_seq_len)

        items = edict(
            tokens=wp_tokens,
            token_ids=wp_token_ids,
            tokens_internal=self.convert_ids_to_tokens(wp_token_ids),
            token_map=wp_map,
            original_tokens=original_tokens,
        )
        print("original tokens: {} \nwp tokens: {} \n internal tokens: {}".format(
            items.original_tokens, items.tokens, items.tokens_internal))
        return items


def get_original_token_embedding(padded_wp_token_embedding, wp_token_mask, token_map):
    """ subword embeddings from the same original token will be averaged to get
    the original token embedding.
    Args:
        padded_wp_token_embedding: (#wp_tokens, hsz)
        wp_token_mask: (#wp_tokens, )
        token_map (list of int): maps the word piece tokens to original tokens

    Returns:

    """
    unpadded_wp_embedding = padded_wp_token_embedding[:int(sum(wp_token_mask).item())]
    original_token_embedding = [unpadded_wp_embedding[token_map[i]:token_map[i+1]].mean(0)
                                for i in range(len(token_map)-1)]
    return np.stack(original_token_embedding)  # (#ori_tokens, hsz)


def decode_b64_to_np_array(b64_str, last_dim_size=768, dtype=np.float32):
    """decode 2d np array from b64 string"""
    def correct_b64_padding(b64_string):
        return b64_string + "=" * ((4 - len(b64_string) % 4) % 4)

    try:
        array = np.frombuffer(base64.decodebytes(b64_str), dtype=dtype).reshape(-1, last_dim_size)
    except TypeError as e:
        b64_str = correct_b64_padding(b64_str)
        array = np.frombuffer(base64.decodebytes(b64_str), dtype=dtype).reshape(-1, last_dim_size)
    return array


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    # Other parameters
    # parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using uncased model.")
    # parser.add_argument("--layers", default="-2", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # layer_indexes = [int(x) for x in args.layers.split(",")]
    layer_index = -2  # second-to-last, which showed reasonable performance in BERT paper

    dset = BertSingleSeqDataset(args.input_file, args.bert_model, args.max_seq_length)
    model = BertModel.from_pretrained(args.bert_model)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    eval_sampler = SequentialSampler(dset)
    eval_dataloader = DataLoader(dset, sampler=eval_sampler, batch_size=args.batch_size,
                                 collate_fn=pad_collate, num_workers=8)

    model.eval()
    torch.set_grad_enabled(False)
    with h5py.File(args.output_file, "w") as h5_f:
        for batch in tqdm(eval_dataloader):
            input_ids = batch.token_ids.to(device)
            input_mask = batch.token_ids_mask.to(device)
            unique_ids = batch.unique_id

            all_encoder_layers, _ = model(input_ids,
                                          token_type_ids=None,
                                          attention_mask=input_mask)  # (#layers, bsz, #tokens, hsz)
            layer_output = all_encoder_layers[layer_index].detach().cpu().numpy()  # (bsz, #tokens, hsz)
            print("layer_output", layer_output.shape)

            for batch_idx, unique_id in enumerate(unique_ids):
                original_token_embeddings = get_original_token_embedding(layer_output[batch_idx],
                                                                         batch.token_ids_mask[batch_idx],
                                                                         batch.token_map[batch_idx])
                h5_f.create_dataset(str(unique_id), data=original_token_embeddings, dtype=np.float32)


if __name__ == "__main__":
    main()





