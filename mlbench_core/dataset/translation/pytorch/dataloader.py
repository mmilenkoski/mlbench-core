import os
import torch
import torchtext
import torchtext.datasets as nlp_datasets
from mlbench_core.dataset.translation.pytorch import config, Tokenizer


def _get_nmt_text(batch_first):
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


class WMT14Dataset(nlp_datasets.WMT14):
    """WMT14 Dataset.

    Loads WMT14 dataset.
    Based on `torchtext.datasets.WMT14`

    Args:
        root (str): Root folder of WEMT14 dataset (without `train/` or `val/`)
        train (bool): Whether to get the train or validation set (default=True)
    """

    def __init__(
        self,
        root,
        tokenizer=None,
        download=True,
        train=True,
        fields=None,
        batch_first=False,
        max_sent_length=150,
    ):
        self.train = train

        self.fields = fields
        if not self.fields:
            self.fields = _get_nmt_text(batch_first=batch_first)

        self.root = root

        if download:
            path = self.download(root)
        else:
            path = os.path.join(root, "wmt14/wmt14")

        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(os.path.join(path, config.VOCAB_FNAME))

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
