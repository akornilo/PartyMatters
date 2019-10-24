
# Word Embedding Dim
k_word = 50

# Number of words to pick per document
n_word_summary = 400
n_word_fulltext = 2000

# Legislator Embedding size
k_leg_emb = 25

# Basic CNN setting
filter_size = 4
num_maps = 600

## Various dropouts
# Added after leg embedding created 
dropout_leg = 0.25

# Dropout bill - added after text embed is created
dropout_bill = 0.3

#dropout added after bill is scaled down to leg space
dropout_bill2 = 0.2