class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.vocab_size = None
        self.text = None

    def _get_stats(self):
        """
        Get the statistics of the text.
        """
        lookup_dict = {}
        for pair in zip(self.text, self.text[1:]):
            lookup_dict[pair] = lookup_dict.get(pair, 0) + 1

        max_pair = max(lookup_dict.items(), key=lambda x: x[1])

        return max_pair

    def _merge_vocab(self, tokens_to_merge, ids):
        """
        Merge the vocabulary.
        """
        self.text = self.text.replace(tokens_to_merge, ids)

    def train(self, text, vocab_size):
        """
        Train the tokenizer using BPE algorithm.
        Params:
            text (str): string-type data used to run BPE.
            vocab_size (int): the size of final vocabulary.

        Return:
            None
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be greater than 256")
        if len(text) < 2:
            raise ValueError("text must be at least 2 characters long")
        self.vocab_size = vocab_size
        self.text = text.encode('utf-8')
        
        while len(self.vocab) < vocab_size and len(self.text) > 1:
            tokens_to_merge, ids = self._get_stats(self.text)
            self._merge_vocab(tokens_to_merge, ids)


    def encode(self, text):
        """
        Encode the input string into a token list.
        Params:
            text (str): data to be tokenized.

        Return:
            ids (list): list of integer-type tokens.
        """
        pass

    def decode(self, ids):
        """
        Decode a token list into a string.
        Params:
            ids (list): list of integer-type tokens.

        Return:
            text (str): string-type data.
        """
        pass
