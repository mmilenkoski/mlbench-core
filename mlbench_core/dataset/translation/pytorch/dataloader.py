import os

import torchtext
import torchtext.datasets as nlp_datasets
from mlbench_core.dataset.translation.pytorch import config, WMT14Tokenizer


def _get_nmt_text(batch_first=False, include_lengths=False, tokenizer="spacy"):
    """ Returns the text fields for NMT

    Args:
        batch_first:

    Returns:

    """
    SRC_TEXT = WMT14Tokenizer(
        language="en",
        tokenizer=tokenizer,
        init_token=config.BOS_TOKEN,
        eos_token=config.EOS_TOKEN,
        pad_token=config.PAD_TOKEN,
        unk_token=config.UNK_TOKEN,
        batch_first=batch_first,
        include_lengths=include_lengths
    )

    TGT_TEXT = WMT14Tokenizer(
        language="de",
        tokenizer=tokenizer,
        init_token=config.BOS_TOKEN,
        eos_token=config.EOS_TOKEN,
        pad_token=config.PAD_TOKEN,
        unk_token=config.UNK_TOKEN,
        batch_first=batch_first,
        include_lengths=include_lengths
    )

    return SRC_TEXT, TGT_TEXT


class WMT14Dataset(nlp_datasets.WMT14):
    def __init__(
            self,
            root,
            download=True,
            train=True,
            batch_first=False,
            include_lengths=False,
            max_size=None,
            max_sent_length=150,
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
        """
        self.train = train
        self.batch_first = batch_first
        self.fields = _get_nmt_text(batch_first=batch_first,
                                    include_lengths=include_lengths)
        self.root = root

        if download:
            path = self.download(root)
        else:
            path = os.path.join(root, "wmt14/wmt14")

        for i in self.fields:
            i.build_vocab_from_file(os.path.join(path, config.VOCAB_FNAME),
                                    max_size=max_size)

        if train:
            path = os.path.join(path, config.TRAIN_FNAME)
        else:
            path = os.path.join(path, config.VAL_FNAME)

        filter_pred = lambda x: not (
                len(vars(x)["src"]) > max_sent_length
                or len(vars(x)["trg"]) > max_sent_length
        )
        super(WMT14Dataset, self).__init__(
            path=path, fields=self.fields, exts=config.EXTS,
            filter_pred=filter_pred
        )

    @property
    def vocab_size(self):
        return self.fields['src'].vocab_size

    def get_raw_item(self, idx):
        return super().__getitem__(idx)

    def get_loader(
            self,
            batch_size=1,
            shuffle=False,
            device=None,

    ):

        train_iter = torchtext.data.BucketIterator(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            device=device,
            sort_key=lambda x: torchtext.data.interleave_keys(len(x.src),
                                                              len(x.trg)))
        return train_iter
