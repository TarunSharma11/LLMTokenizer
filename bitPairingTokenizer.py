class BitPairingTokenizer():
    def __init__(self) -> None:
        self.merges = None
        self.vocab = None
    
    def __get_stats(self, encoding):
        freq_map = {}
        for p1, p2 in zip(encoding[:-1], encoding[1:]):
            freq_map[(p1,p2)] = freq_map.get((p1,p2), 0) + 1
        return freq_map
    
    def __merge_pair(self, encoding, candidate_pair, merge_value):
        merged_encoding = []
        i = 0
        while(i < len(encoding)):
            if( (i < (len(encoding) - 1)) and (encoding[i] == candidate_pair[0]) and (encoding[i+1] == candidate_pair[1])):
                merged_encoding.append(merge_value)
                i = i + 2
            else:
                merged_encoding.append(encoding[i])
                i = i + 1
        return merged_encoding
    
    def __create_merges(self, encoding, num_of_merges):
        self.merges = {}
        for m in range(num_of_merges):
            freq = self.__get_stats(encoding)
            candidate_pair = max(freq, key = freq.get)
            if (freq[candidate_pair] == 1):
                break
            merge_value = 256 + m
            encoding = self.__merge_pair(encoding, candidate_pair, merge_value)
            self.merges[candidate_pair] = merge_value
        return encoding
    
    def __create_vocab(self, merges):
        if(merges is None):
            raise Exception("Please create merges before vocabulary")
        self.vocab = {i: bytes([i]) for i in range(256)}
        for m in merges:
            self.vocab[merges[m]] = self.vocab[m[0]] + self.vocab[m[1]]

    def train(self, text, vocab_size, verbose = False):
        encoded_text = text.encode("utf-8")
        encoding = list(map(int, encoded_text))
        num_of_merges = vocab_size - 256
        encoding = self.__create_merges(encoding, num_of_merges)
        self.__create_vocab(self.merges)
    
    def encode(self, text):
        if(self.merges is None or self.vocab is None):
            raise Exception("Please train your model using the train method before encoding")
        encoding = text.encode('utf-8')
        for m in self.merges:
            encoding = self.__merge_pair(encoding, m, self.merges[m])
        return encoding
                    
    def decode(self, ids):
        if(self.merges is None or self.vocab is None):
            raise Exception("Please train your model using the train method before decoding")
        decoded_bytes = b"".join(self.vocab[id] for id in ids)
        decoded_string = decoded_bytes.decode('utf-8', errors= 'replace')
        return decoded_string
    
if __name__ == '__main__':
    tk = BitPairingTokenizer()

    with open("trainText.txt", encoding = 'utf-8') as f:    
        text = f.read()
    tk.train(text, 276)   
    print(text == tk.decode(tk.encode(text)))
