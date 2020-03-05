import torchtext
from mlbench_core.dataset.translation.pytorch import config


class WMT14Tokenizer(torchtext.data.Field):

    def __init__(self, language, tokenizer, separator="@@", **kwargs):
        """ Tokenizer Class for WMT14 that uses the whole vocabulary

        Args:
            language (str): One of `en` or `de`
            tokenizer (str): Tokenizer class, usually `spacy`
            separator (str): Separator for words
            **kwargs: Other arguments to be passed to `torchtext.data.Field`
        """
        self.separator = separator

        tokenize = torchtext.data.get_tokenizer(tokenizer, language=language)

        super(WMT14Tokenizer, self).__init__(tokenize=tokenize, **kwargs)

    def build_vocab_from_file(self, vocab_fpath, max_size=None):
        """
        Builds The vocabulary from a given filename
        Args:
            vocab_fpath (str): Vocabulary path
            max_size (str): Max Vocab size

        """
        vocab = []
        with open(vocab_fpath) as vfile:
            for line in vfile:
                vocab.append([line.strip()])

        super(WMT14Tokenizer, self).build_vocab(vocab, max_size=max_size)

    def detokenize(self, inputs, delim=" "):
        """
        Detokenizes single sentence and removes token separator characters.

        Args:
            inputs (str): Input sequences
            delim (str): Tokenization delimiter

        Return:
            String representing detokenized sentence
        """
        detok = delim.join([self.vocab.itos[idx] for idx in inputs])
        detok = detok.replace(self.separator + " ", "")
        detok = detok.replace(self.separator, "")

        detok = detok.replace(config.BOS_TOKEN, "")
        detok = detok.replace(config.EOS_TOKEN, "")
        detok = detok.replace(config.PAD_TOKEN, "")
        detok = detok.strip()
        return detok

    @property
    def vocab_size(self):
        return len(self.vocab.itos)
