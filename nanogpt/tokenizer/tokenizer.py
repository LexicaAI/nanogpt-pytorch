import tiktoken

class Tokenizer:
    """
    Wrapper around tiktoken tokenizers
    """
    def __init__(self, encoding_name="gpt2"):
        self.encoding_name = encoding_name
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.eot = self.encoder._special_tokens['<|endoftext|>']  # end of text token
    
    def encode(self, text):
        """
        Encodes the given text to token IDs.
        """
        return self.encoder.encode(text)
    
    def encode_ordinary(self, text):
        """
        Encodes the given text to token IDs without special tokens.
        """
        return self.encoder.encode_ordinary(text)
    
    def decode(self, token_ids):
        """
        Decodes the given token IDs back to text.
        """
        return self.encoder.decode(token_ids)
    
    def get_vocab_size(self):
        """
        Returns the vocabulary size.
        """
        return self.encoder.n_vocab 