class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.vocab_size = 0
        self.text = []

    def _get_stats(self):
        """
        Get the statistics of the text.
        """
        pair_count = {}
        for pair in zip(self.text, self.text[1:]):
            pair_count[pair] = pair_count.get(pair, 0) + 1

        return pair_count

    def _merge_vocab(self, tokens_to_merge, i):
        """
        Merge the vocabulary.
        """
        self.vocab[i] = (tokens_to_merge[0], tokens_to_merge[1])
        self.inv_vocab[(tokens_to_merge[0], tokens_to_merge[1])] = i
        j = 0
        while j < len(self.text) - 1:
            if self.text[j] == tokens_to_merge[0] and self.text[j + 1] == tokens_to_merge[1]:
                self.text[j] = i
                del self.text[j + 1]
            j += 1

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
        self.text = list(text.encode('utf-8'))
        self.vocab.clear()
        self.inv_vocab.clear()
        
        i = 256
        while i < vocab_size and len(self.text) > 1:
            pair_count = self._get_stats()
            max_pair = max(pair_count.items(), key=lambda x: x[1])[0]
            self._merge_vocab(max_pair, i)
            i += 1


    def encode(self, text):
        """
        Encode the input string into a token list.
        Params:
            text (str): data to be tokenized.

        Return:
            ids (list): list of integer-type tokens.
        """
        self.text = list(text.encode('utf-8'))

        while len(self.text) > 1:
            pairs = set()
            min_pair_ids = self.vocab_size
            for pair in zip(self.text, self.text[1:]):
                if pair in self.inv_vocab:
                    pairs.add(self.inv_vocab[pair])
                    min_pair_ids = min(min_pair_ids, self.inv_vocab[pair])
            if min_pair_ids == self.vocab_size:
                break
            min_pair = self.vocab[min_pair_ids]
            
            j = 0
            while j < len(self.text) - 1:
                if self.text[j] == min_pair[0] and self.text[j + 1] == min_pair[1]:
                    self.text[j] = min_pair_ids
                    del self.text[j + 1]
                j += 1

        return self.text
        

    def decode(self, ids):
        """
        Decode a token list into a string.
        Params:
            ids (list): list of integer-type tokens.

        Return:
            text (str): string-type data.
        """
        if len(ids) == 0:
            return ""

        i = 0
        while i < len(ids):
            if ids[i] < 256:
                i += 1
            else:
                pair = self.vocab[ids[i]]
                ids[i] = pair[0]
                ids.insert(i + 1, pair[1])

        text = bytes(ids).decode('utf-8')
        return text

if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokenizer.train("hello world", 270)
    print(list("hello world".encode('utf-8')))
    print(tokenizer.text)
    print(tokenizer.encode("hello world"))
    print(tokenizer.decode(tokenizer.encode("hello world")))