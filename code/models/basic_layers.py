from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.layers import Embedding, Input, SpatialDropout1D, Reshape, Lambda, Dropout, Dense, Multiply, Conv1D, Flatten, GRU, MaxPooling2D, Conv2D
from keras.models import Sequential, Model
#from keras.metrics import fmeasure
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold

from feature_processing.preprocess_text import prepare_text_features
from feature_processing.word_embeddings import parse_glove_embeddings
from models.default_constants import *
from utilities.metrics import precision, recall

# disable debugging messages in TF
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def mask_aware_mean(x):
    # Special method to only average non-zero embeddings and avoid padding
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)

    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)

    # compute mask-aware mean of x
    x_mean = K.sum(x, axis=1, keepdims=False) / n

    return x_mean

def mask_aware_mean_output_shape(input_shape):
    # Support method for mask_aware_mean - needed to complete layer
    shape = list(input_shape)
    assert len(shape) == 3 
    return (shape[0], shape[2])

def get_basic_leg_embedding(leg_input, k_leg_emb=25, n_leg=550, dropout_leg=0.25, **kwargs):  
    '''
    Create basic legislator embeddings

    leg_input: Previous layer to connect to legislator embedding
    k_leg_emb: dimension of leg embedding 
    n_leg: total number of legislators layer
    dropout_leg: how much dropout to add in between layers
    '''

    leg_embed = Embedding(input_dim=n_leg, output_dim=k_leg_emb, input_length=1,
                        embeddings_initializer='glorot_uniform',
                        name='leg_embedding')(leg_input)

    leg_embed = SpatialDropout1D(dropout_leg)(leg_embed)

    flat_leg_embed = Reshape((k_leg_emb, ), name='final_leg_layer')(leg_embed)

    return flat_leg_embed

def get_basic_bill_embedding(bill_input, num_word_embed, k_word=50, n_word=250,  
                             dropout_bill=0.3,  weights=None, 
                             layer_prefix='text', **kwargs):
    '''
    Simple bill embedding that averages the words in it. 
    Returns the final layer's output in this sequence
    
    bill_input: An input layer for text.

    vocab_size: Number of distinct possible words in input
    k_word: dimension of word embeddings
    n_word: length of input bills in words 
    dropout_bill: How much dropout to use after embeddings
    weights: predefined embedding weights, otherwise use random (glorot_uniform)
    layer_prefix: prefix to use for layers in here (useful when many diff texts)
    '''

    if weights is not None:
        bill_embed = Embedding(input_dim=weights.shape[0],
                output_dim=k_word,
                input_length=n_word,
                weights=[weights],
                trainable=True,
                name='embed_' + layer_prefix
                )(bill_input)
    else:
        bill_embed = Embedding(input_dim=num_word_embed,
                output_dim=k_word,
                input_length=n_word,
                trainable=True,
                embeddings_initializer='glorot_uniform',
                name='embed_' + layer_prefix
                )(bill_input)

    bill_embed = SpatialDropout1D(dropout_bill, name='dropout_' + layer_prefix)(bill_embed)

    # Average non-zero embeddings only
    average_bill_embed = Lambda(mask_aware_mean, mask_aware_mean_output_shape, 
                                 name='avg_{}'.format(layer_prefix))(bill_embed)

    return average_bill_embed


def mask_aware_mean_output_shape(input_shape):
    # Support method for mask_aware_mean - needed to complete layer
    shape = list(input_shape)
    assert len(shape) == 3 
    return (shape[0], shape[2])

def get_cnn_bill_embedding(bill_input, num_word_embed, k_word=50, n_word=250,  
                             weights=None, dropout_bill=0.3, dropout_cnn=0.3,
                             layer_prefix='text', filter_size=4, num_maps=600,
                             **kwargs):
    '''
    Simple bill embedding that uses an CNN. 
    Returns the final layer's output in this sequence
    
    bill_input: An input/previous layer for text.

    vocab_size: Number of distinct possible words in input
    k_word: dimension of word embeddings
    n_word: length of input bills in words 
    weights: pre-defined weights for embedding layer 
    layer_prefix: prefix to use for all bill layers (relevant when many texts present)
    filter_size: size of the cnn filter 
    num_maps: number of filters to use
    dropout_bill: Dropout to use after bill later
    '''

    if weights is not None:
        bill_embed = Embedding(input_dim=num_word_embed,
                output_dim=k_word,
                input_length=n_word,
                weights=[weights],
                trainable=True,
                name=layer_prefix + '_embed'
                )(bill_input)
    else:
        bill_embed = Embedding(input_dim=num_word_embed,
                output_dim=k_word,
                input_length=n_word,
                trainable=True,
                embeddings_initializer='glorot_uniform',
                name=layer_prefix + '_embed'
                )(bill_input)

    bill_embed = SpatialDropout1D(dropout_bill, name=layer_prefix+'_dropoutbill')(bill_embed)
    # Add additional dimensions (needed to work with cnn)
    bill_embed = Reshape((n_word, k_word, 1))(bill_embed)

    # The CNN Magic
    cnn_bill_embed = Conv2D(num_maps, (filter_size, 50),  padding='valid', 
                            activation='relu', kernel_initializer='normal',
                            name=layer_prefix + '_cnn')(bill_embed)
    cnn_bill_embed = MaxPooling2D((250 - filter_size + 1, 1), 
                                    name=layer_prefix+'_cnn_maxpool')(cnn_bill_embed)

    final_bill_embed = Flatten()(cnn_bill_embed)

    final_bill_embed = Dropout(dropout_cnn, name=layer_prefix+'_cnn_dropout')(final_bill_embed)

    return final_bill_embed

def get_basic_model(n_leg, num_word_embed, bill_type='basic', n_word=250,
                    k_leg_emb=10, dropout_bill2=0.2, **kwargs):
    '''
    The Basic model for predicting votes. Takes in a legislator and a bill text,
    creates a bill representation based on the text, transforms both to the same 
    space and takes a 'dot product'.

    n_leg: Total number of legislators in data set 
    num_word_embed: size of the text vocabulary
    bill_type: basic or cnn - what type of text embed to create
    n_word: length of input texts 
    k_leg_emb: size of leg embeddings 
    dropout_bill2: dropout added after bill is scaled down to leg space
    kwargs: additional arguments for layers
    '''

    # Create bill and legislator embeddings
    leg_input = Input(shape=(1, ), dtype="int32", name="leg_input")
    leg_embed = get_basic_leg_embedding(leg_input, k_leg_emb, n_leg, dropout_leg)

    bill_input = Input(shape=(n_word,), dtype="int32", name="bill_input")

    if bill_type == 'cnn':
        bill_embed = get_cnn_bill_embedding(bill_input, num_word_embed, 
                                            n_word=n_word, **kwargs)
    else:
        bill_embed = get_basic_bill_embedding(bill_input, num_word_embed, 
                                                n_word=n_word, **kwargs)

    # Transform bill to legislator space
    bill_scaled = Dense(k_leg_emb, activation='linear')(bill_embed)
    bill_scaled = Dropout(dropout_bill2)(bill_scaled)
    
    # Combine the bill and leg embeddings
    bill_leg_concat = Multiply()([bill_scaled, leg_embed])
    bill_leg_product = Dense(k_leg_emb, activation='linear', name='product_scaler')(bill_leg_concat)

    # Make prediction
    main_output = Dense(1, activation='sigmoid', name='main_output')(bill_leg_product)
    model = Model(inputs=[leg_input, bill_input], outputs=[main_output])

    model.compile(loss={'main_output': 'binary_crossentropy'},
                  optimizer='adamax', metrics=['accuracy', precision, recall])

    print(model.summary())

    return model


def prepare_basic_model_features(vote_df, text_field='text',
                                 n_word=250, unique_words=False, vocab_size=10000, 
                                 vectorizer=None, leg_crosswalk=None, **kwargs):
    '''
    Set-up the text and legislator id fields for given data set. Uses passed 
    in id maps, if None, creates new. For text, maps words to ids, for legislators
    maps natural_ids to numeric ids.
    Returns processed dataset + maps to convert text/legs to ids (vectorizer/leg_crosswalk)
        The maps will either the ones passed in OR new ones if we had to create them.

    vote_df: Pandas dataframes with text and legislators 
    text_field: specifies which columns in pandas frames, contains version of text 
        that we want to process for features.
    n_word: maximum vocabulary size to use
    unique_words: whether the text vectorizer should consider unique words or the 
        sequence of all words when converting to ids.
    vocab_size: maximum vocabulary size during vectorizing (can be None or int)
    vectorizer: predefined vectorizer for both train/test 
    leg_crosswalk: map from natural_ids to int ids (if predefined)
    '''

    # Set-up common system to map both train and test data to 
    # Only create a new one, if it has not already been defined.
    if leg_crosswalk is None:
        leg_ids = vote_df["leg_id"].unique()
        leg_crosswalk = pd.Series(leg_ids).to_dict()
    
    leg_crosswalk_rev = dict((v, k) for k, v in leg_crosswalk.items())
    vote_df['leg_id_v2'] = vote_df.leg_id.map(leg_crosswalk_rev)

    # If some legislators did not get a leg_id, drop them
    vote_df = vote_df[vote_df.leg_id_v2.notnull()]

    # Prepare text data using same vectorizer features
    text_df, vectorizer = prepare_text_features(vote_df, max_words=n_word, 
                                                unique_words=unique_words, 
                                                text_field=text_field, 
                                                vocab_size=vocab_size, 
                                                vectorizer=vectorizer)

    all_data = [vote_df["leg_id_v2"].values, text_df]

    return all_data, vectorizer, leg_crosswalk

def get_default_callbacks(experiment_name='some_model', patience=10, min_delta=0,
            use_checkpoint=True, use_early_stop = True):

    save_file = '../model_files/' + experiment_name

    callbacks = []

    if use_early_stop:
        callbacks.append(EarlyStopping('val_loss', patience=patience, min_delta=min_delta))

    if use_checkpoint:
        callbacks.append(ModelCheckpoint(save_file, save_best_only=True, save_weights_only=False))

    return callbacks

def batch_generator(features, labels, batch_size=50):
 # Create empty arrays to contain batch of features and labels#

    while True:
        indicies = random.sample(range(0, len(labels)), batch_size)
        batch_features = [feats[indicies] for feats in features]
        batch_labels = labels[indicies]
        yield batch_features, batch_labels

def evaluate_model(vote_df_train, vote_df_test, 
                    experiment_name='some_experiment', **kwargs):

    all_train_data, vectorizer, leg_crosswalk = prepare_basic_model_features(vote_df_train,
                                                  **kwargs)

    all_test_data, _, _ = prepare_basic_model_features(vote_df_test,
                                                  vectorizer=vectorizer, 
                                                  leg_crosswalk=leg_crosswalk,
                                                  **kwargs)

    all_2013, _, _ = prepare_basic_model_features(vote_df_2013, 
                                                  vectorizer=vectorizer, 
                                                  leg_crosswalk=leg_crosswalk,
                                                  **kwargs)

    all_2015, _, _ = prepare_basic_model_features(vote_df_2015, 
                                                  vectorizer=vectorizer, 
                                                  leg_crosswalk=leg_crosswalk,
                                                  **kwargs)

    # Load in the word_embeddings
    word_embeddings = parse_glove_embeddings(vectorizer.vocabulary_)

    # Get distinct legs in train data
    n_leg = len(vote_df_train.leg_id_v2.unique())
    vocab_size = word_embeddings.shape[0]

    train_votes = vote_df_train["vote"].astype(int).values
    test_votes = vote_df_test["vote"].astype(int).values

    callbacks = get_default_callbacks(experiment_name=experiment_name, use_checkpoint=True)

    model = get_basic_model(n_leg, vocab_size, word_embeddings=word_embeddings, 
                            **kwargs)

    model.fit_generator(batch_generator(all_train_data, train_votes),
          validation_data=(all_test_data,  test_votes),
          steps_per_epoch=len(train_votes) / 50,
          epochs=5,
          verbose=1,
          class_weight={0: sample_weights[0],
                                  1: sample_weights[1]},
          callbacks=callbacks)
    
    score = model.evaluate(all_test_data,  test_votes)

    print('2013', model.evaluate(all_2013, vote_df_2013.vote))

    print('2015', model.evaluate(all_2015, vote_df_2015.vote))

    return score


if __name__ == '__main__':
    prefix = "../data/us_data/"

    mode = 'summary' # 'fulltext'
    bill_type = 'cnn' # 'cnn'
    experiment_name = '{}_{}_v1'.format(mode, bill_type)

    if mode == 'summary':
        n_word = n_word_summary
    else:
        n_word = n_word_fulltext

    vote_df_all = pd.read_pickle(prefix + 'full_20052012_99_{}.pkl'.format(mode))
    all_bill_ids = vote_df_all.natural_id.unique()

    vote_df_2013 = pd.read_pickle(prefix + 'full_20132014_99_{}.pkl'.format(mode))
    vote_df_2015 = pd.read_pickle(prefix + 'full_20152016_99_{}.pkl'.format(mode))
   
    # ---- Generic baseline stuff --- #
    for years, votes in [('2005-2012', vote_df_all), ('2013-2014', vote_df_2013), 
                        ('2015-2016', vote_df_2015)]:
    
        ys = votes["vote"].astype(int).values
        sample_weights = (1.0 * ys.shape[0]) / (len(np.unique(ys)) * np.bincount(ys))
        print("Baseline Accuracy {}".format(years), 1. * sum(ys) / len(ys))

    # Let's cross-validate some data
    kf = KFold(5)
    all_scores = []
    i = 0
    for train_ids, test_ids in kf.split(all_bill_ids):
        print("On Partition:", i)
        i += 1

        train_bills = all_bill_ids[train_ids]

        print(train_bills[0])

        vote_df_train = vote_df_all[vote_df_all.natural_id.isin(train_bills)]
        vote_df_test = vote_df_all[~vote_df_all.natural_id.isin(train_bills)]

        score = evaluate_model(vote_df_train, vote_df_test, bill_type=bill_type,
                                experiment_name=experiment_name,text_field='special_text',
                                n_word=n_word, vocab_size=10000,
                                dropout_leg=dropout_leg, dropout_bill=dropout_bill,
                                dropout_bill2=dropout_bill2)

        all_scores.append(score)

    pickle.dump(all_scores, open('basic_model_score-new.pkl', 'wb'))
