import regex as re
import tiktoken

class TuneGPT4Tokenizer():
    """
    When we tune LLM they might not perform as well the underlying tokenizer might not have the required tokens for the new use case.
    This class is to add new words and tokens to the vocab of gpt-4 tokenizer (can easily be extended to other LLMs). 
    
    NOTE: this is work in progress.   
    """
    def __init__(self) -> None:
        self.regex_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.regex_build = re.compile(self.regex_pattern)
        enc = tiktoken.get_encoding('cl100k_base')
        self.mergeable_ranks = enc._mergeable_ranks
        self.vocab = {v:k for k,v in self.mergeable_ranks.items()}
        self.merges = self.recover_merges(self.mergeable_ranks)
        self.GPT4_SPECIAL_TOKENS = {
            '<|endoftext|>': 100257,
            '<|fim_prefix|>': 100258,
            '<|fim_middle|>': 100259,
            '<|fim_suffix|>': 100260,
            '<|endofprompt|>': 100276
        }
        
    def bpe(self, mergeable_ranks, token, max_rank):
        # helper function used in get_gpt4_merges() to reconstruct the merge forest
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        return parts
    
    def __merge_pair(self, encoded_words, candidate_pair, merge_value):
        merged_encoding = []
        for word in encoded_words:
            merged_word = []
            i = 0
            while(i < len(word)):
                if( (i < (len(word) - 1)) and (word[i] == candidate_pair[0]) and (word[i+1] == candidate_pair[1])):
                    merged_word.append(merge_value)
                    i = i + 2
                else:
                    merged_word.append(word[i])
                    i = i + 1
            merged_encoding.append(merged_word)
        return merged_encoding
        
    def recover_merges(self, mergeable_ranks):
        merges = {}
        for token, rank in mergeable_ranks.items():
            if len(token) == 1:
                continue # skip raw bytes
            pair = tuple(self.bpe(mergeable_ranks, token, max_rank=rank))
            assert len(pair) == 2
            # recover the integer ranks of the pair
            ix0 = mergeable_ranks[pair[0]]
            ix1 = mergeable_ranks[pair[1]]
            merges[(ix0, ix1)] = rank

        return merges        
            
    def encode(self, text):
        if(self.merges is None or self.vocab is None):
            raise Exception("Please train your model using the train method before encoding")
        word_list = re.findall(self.regex_build, text)
        encoded_words = [list(map(lambda x: self.mergeable_ranks[bytes([x])], w.encode('utf-8'))) for w in word_list]
        for m in self.merges:
            encoded_words = self.__merge_pair(encoded_words, m, self.merges[m])
        merged_words = []
        for word in encoded_words:
            merged_words.extend(word)
        return merged_words
                    
    def decode(self, ids):
        if(self.merges is None or self.vocab is None):
            raise Exception("Please train your model using the train method before decoding")
        decoded_bytes = b"".join(self.vocab[id] for id in ids)
        decoded_string = decoded_bytes.decode('utf-8', errors= 'replace')
        return decoded_string
    
    def encode_special(self, text):
        special_pattern = "(" + "|".join(re.escape(x) for x in self.GPT4_SPECIAL_TOKENS) + ")"
        split_sentences = re.split(special_pattern, text)
        encoding = []
        for sentence in split_sentences:
            if(sentence in self.GPT4_SPECIAL_TOKENS):
                encoding.append(self.GPT4_SPECIAL_TOKENS[sentence])
            else:
                encoding.extend(self.encode(sentence))
        return encoding
    
    def decode_special(self, ids):
        inv_special_tokens = {v:k for k,v in self.GPT4_SPECIAL_TOKENS.items()}
        decoded_bytes = b""
        for id in ids:
            if id in inv_special_tokens:
                decoded_bytes += inv_special_tokens[id].encode('utf-8')
            else:
                decoded_bytes += self.vocab[id]
        return decoded_bytes.decode('utf-8', errors='replace')

    