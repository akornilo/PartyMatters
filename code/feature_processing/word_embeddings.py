import numpy as np

def parse_glove_embeddings(word_dict, k_word=50, embed_file='../data/glove.6B.50d.txt'):
    '''
    Map words to pre-trained embeddings. OOV words will be initialized to 0s 

    word_dict: dictionary mapping from words to indicies.
    embed_file: Link to Glove embeddings file.
    
    Returns matrix, where each row corresponds to an embedding.
    '''

    print("Parsing Embedding Matrix")
    embeddings = np.zeros((len(word_dict) + 1, k_word))
    included_words = {}
    with open(embed_file, 'r') as f:
        content = f.read().split('\n')
        for line in content:
            data = line.split(' ')
            if data[0] in word_dict:
                embeddings[word_dict[data[0]] + 1] = list(map(float, data[1:]))
                included_words[data[0]] = 1
    for word in word_dict:
        if word not in included_words:
            # print word
            embeddings[word_dict[word] + 1] = np.zeros(k_word)
    print(embeddings.shape)
    return np.array(embeddings, dtype=np.float)
