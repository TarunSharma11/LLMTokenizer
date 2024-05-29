import regex as re

class GPT4Tokenizer():
    def __init__(self) -> None:
        self.regex_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.regex_build = re.compile(self.regex_pattern)
        
    def __calc_stats(self, encoded_words):
        stats = {}
        for word in encoded_words:
            for p1,p2 in zip(word[1:], word[:-1]):
                stats[(p1,p2)] = stats.get((p1,p2), 0) + 1
        return stats
    
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
                
    def __create_vocab(self, merges):
        if(merges is None):
            raise Exception("Please create merges before vocabulary")
        self.vocab = {i: bytes([i]) for i in range(256)}
        for m in merges:
            self.vocab[merges[m]] = self.vocab[m[0]] + self.vocab[m[1]]
            
    def train(self, text, vocab_size, verbose = False):
        word_list = re.findall(self.regex_build, text)
        encoded_words = [list(map(int, w.encode('utf-8'))) for w in word_list]
        num_of_merges = vocab_size - 256
        encoded_words = self._create_merges(encoded_words, num_of_merges)
        self.__create_vocab(self.merges)

            
    def _create_merges(self, encoded_words, num_of_merges):
        self.merges = {}
        for m in range(num_of_merges):
            stats = self.__calc_stats(encoded_words)
            candidate_pair = max(stats, key = stats.get)
            if(stats[candidate_pair] == 1):
                break
            merge_value = 256 + m
            encoded_words = self.__merge_pair(encoded_words, candidate_pair, merge_value)
            self.merges[candidate_pair] = merge_value
        return encoded_words
        
    
    def encode(self, text):
        if(self.merges is None or self.vocab is None):
            raise Exception("Please train your model using the train method before encoding")
        word_list = re.findall(self.regex_build, text)
        encoded_words = [list(map(int, w.encode('utf-8'))) for w in word_list]
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
    
if __name__ == '__main__':
    tk = GPT4Tokenizer()

    with open("trainText.txt", encoding = 'utf-8') as f:    
        text = f.read()
    tk.train(text, 276)   
    print(text == tk.decode(tk.encode(text)))
