import os

import torchtext
import torchtext.datasets as nlp_datasets

PAD_WORD = "<blank>"
UNK_WORD = "<unk>"
BOS_WORD = "<s>"
EOS_WORD = "</s>"
EXTS = (".en", ".de")


def _get_nmt_text(batch_first):
    tokenizer = "spacy"
    SRC_TEXT = torchtext.data.Field(
        tokenize=torchtext.data.utils.get_tokenizer(tokenizer, language="en"),
        pad_token=PAD_WORD,
        batch_first=batch_first,
    )
    TGT_TEXT = torchtext.data.Field(
        tokenize=torchtext.data.utils.get_tokenizer(tokenizer, language="de"),
        init_token=BOS_WORD,
        eos_token=EOS_WORD,
        pad_token=PAD_WORD,
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

        if train:
            path = os.path.join(path, "train.tok.clean.bpe.32000")
        else:
            path = os.path.join(path, "newstest2013.tok.bpe.32000")

        filter_pred = lambda x: not (
            len(vars(x)["src"]) > max_sent_length
            or len(vars(x)["trg"]) > max_sent_length
        )
        super(WMT14Dataset, self).__init__(
            path=path, fields=self.fields, exts=EXTS, filter_pred=filter_pred
        )
