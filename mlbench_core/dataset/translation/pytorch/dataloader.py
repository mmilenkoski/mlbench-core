import os

import torch
import torchtext
import torchtext.datasets as nlp_datasets
from mlbench_core.dataset.translation.pytorch import config, Tokenizer
from torch.utils.data import DataLoader


def collate_seq(seq, batch_first):
    """
    Builds batches for training or inference.
    Batches are returned as pytorch tensors, with padding.

    Args:
        seq (tensor): Sequences
        batch_first (bool): Whether the batch length comes as first dimension

    Returns:
        (tensor, tensor) The sequence as a tensor as well as its length
    """
    lengths = [len(s) for s in seq]
    batch_length = max(lengths)

    shape = (batch_length, len(seq))
    seq_tensor = torch.full(shape, config.PAD, dtype=torch.int64)

    for i, s in enumerate(seq):
        end_seq = lengths[i]
        seq_tensor[:end_seq, i].copy_(s[:end_seq])

    if batch_first:
        seq_tensor = seq_tensor.t()

    return seq_tensor, torch.tensor(lengths)


def parallel_collate(sort, batch_first):
    """
    Builds batches from parallel dataset (src, tgt), optionally sorts batch
    by src sequence length.
    Args:
        sort (bool): Sort by sequence length
        batch_first (bool): Batch length as first dimension

    Returns:
        func
    """

    def _func(seqs):
        src_seqs, tgt_seqs = zip(*seqs)
        if sort:
            indices, src_seqs = zip(
                *sorted(
                    enumerate(src_seqs), key=lambda item: len(item[1]), reverse=True
                )
            )
            tgt_seqs = [tgt_seqs[idx] for idx in indices]

        return tuple(
            [collate_seq(s, batch_first=batch_first) for s in [src_seqs, tgt_seqs]]
        )

    return _func


def _get_nmt_text(batch_first):
    """ Returns the text fields for NMT

    Args:
        batch_first:

    Returns:

    """
    tokenizer = "spacy"
    SRC_TEXT = torchtext.data.Field(
        tokenize=torchtext.data.utils.get_tokenizer(tokenizer, language="en"),
        pad_token=config.PAD_TOKEN,
        batch_first=batch_first,
    )
    TGT_TEXT = torchtext.data.Field(
        tokenize=torchtext.data.utils.get_tokenizer(tokenizer, language="de"),
        init_token=config.BOS_TOKEN,
        eos_token=config.EOS_TOKEN,
        pad_token=config.PAD_TOKEN,
        batch_first=batch_first,
    )
    return SRC_TEXT, TGT_TEXT


def pad_vocabulary(math):
    if math == "fp16":
        pad_vocab = 8
    elif math == "fp32":
        pad_vocab = 1
    else:
        raise NotImplementedError()
    return pad_vocab


class WMT14Dataset(nlp_datasets.WMT14):
    def __init__(
        self,
        root,
        download=True,
        train=True,
        batch_first=False,
        max_sent_length=150,
        math="fp32",
    ):
        """WMT14 Dataset.

        Loads WMT14 dataset.
        Based on `torchtext.datasets.WMT14`

        Args:
            root (str): Root folder of WMT14 dataset
            download (bool): Download dataset
            train (bool): Whether to get the train or validation set.
                Default=True
            batch_first (bool): if True the model uses (batch,seq,feature)
                tensors, if false the model uses (seq, batch, feature)
            max_sent_length (int): Max sentence length
            math (str): One of `fp16` `fp32`, determines vocabulary padding
        """
        self.train = train
        self.batch_first = batch_first
        self.fields = _get_nmt_text(batch_first=batch_first)
        self.root = root

        if download:
            path = self.download(root)
        else:
            path = os.path.join(root, "wmt14/wmt14")

        self.tokenizer = Tokenizer(
            os.path.join(path, config.VOCAB_FNAME), pad_vocabulary(math)
        )

        if train:
            path = os.path.join(path, config.TRAIN_FNAME)
        else:
            path = os.path.join(path, config.VAL_FNAME)

        filter_pred = lambda x: not (
            len(vars(x)["src"]) > max_sent_length
            or len(vars(x)["trg"]) > max_sent_length
        )
        super(WMT14Dataset, self).__init__(
            path=path, fields=self.fields, exts=config.EXTS, filter_pred=filter_pred
        )

    def __getitem__(self, idx):
        example = super().__getitem__(idx)
        src, tgt = example.src, example.trg

        src = torch.tensor(self.tokenizer.segment(src))
        tgt = torch.tensor(self.tokenizer.segment(tgt))

        return src, tgt

    def get_raw_item(self, idx):
        return super().__getitem__(idx)

    def get_loader(
        self,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    ):

        collate_fn = parallel_collate(sort=True, batch_first=self.batch_first)
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
