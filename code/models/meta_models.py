import keras.backend as K
from keras.layers import Embedding, Input, SpatialDropout1D, Reshape, Lambda, Dropout, Dense, Multiply, Add
#from keras.metrics import fmeasure
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adamax
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.model_selection import KFold


from feature_processing.preprocess_text import prepare_text_features
from feature_processing.metafeature_preparation import prepare_sponsor_party_percs
from feature_processing.word_embeddings import parse_glove_embeddings

from models.basic_layers import get_basic_leg_embedding, get_basic_bill_embedding,  get_cnn_bill_embedding, get_default_callbacks, batch_generator, prepare_basic_model_features
from models.default_constants import *

from utilities.metrics import recall, precision

def get_two_party_meta_model(n_leg, num_word_embed,  
                            k_leg_emb=k_leg_emb, n_word=250, word_embeddings=None,
                            bill_type='basic', **kwargs):
    '''
    This version creates a democratic + republican version of the bill, then takes
    a 'dot product'. Takes various parameter settings and returns a Keras Model 

    n_leg: number of total legislators
    num_word_embed: number of words in the bill embedding layer 
    k_leg_emb: size of legislator embedding 
    n_word: length of text input 
    word_embeddings: predefined word weights 
    bill_type: What kinds of bill embeddings to use - text or cnn 
    **kwargs: additional parameters for laters
    '''

    # Create legislator embeddings
    leg_input = Input(shape=(1, ), dtype="int32", name="leg_input")
    leg_embed = get_basic_leg_embedding(leg_input, k_leg_emb, n_leg, **kwargs)

    ##### Create two copies of the text - they will be scaled by party ########
    
    ## Republican sponsored embedding
    bill_input_rep = Input(shape=(n_word,), dtype="int32", name="bill_input_rep")
    if bill_type == 'cnn':
        bill_embed_rep = get_cnn_bill_embedding(bill_input_rep, num_word_embed,   
                                                layer_prefix="rep", n_word=n_word,
                                                **kwargs)
        text_output_dim = 600
    else: 
        bill_embed_rep = get_basic_bill_embedding(bill_input_rep, num_word_embed,
                                                  layer_prefix="rep", n_word=n_word,
                                                  **kwargs)
        text_output_dim = 50
    # Scale the party perc to match the size of the text embed
    rep_perc = Input(shape=(1,), dtype="float32", name='rep_perc_input')
    rep_perc2 = Dense(text_output_dim, activation='linear', name='rep_perc_scale')(rep_perc)
    bill_rep = Multiply(name="rep_scaled")([rep_perc2, bill_embed_rep])

    ## Democrat sponsored embedding
    bill_input_dem = Input(shape=(n_word,), dtype="int32", name="bill_input_dem")
    if bill_type == 'cnn':
        bill_embed_dem = get_cnn_bill_embedding(bill_input_dem, num_word_embed, 
                                                weights=word_embeddings, 
                                                layer_prefix="dem", n_word=n_word,
                                                **kwargs)
        text_output_dim = 600
    else: 
        bill_embed_dem = get_basic_bill_embedding(bill_input_dem, num_word_embed,  
                                                weights=word_embeddings, 
                                                layer_prefix="dem", n_word=n_word,
                                                **kwargs)
        text_output_dim = 50

    ## Scale the party perc to match the size of the text embed
    dem_perc = Input(shape=(1,), dtype="float32", name='dem_perc_input')
    dem_perc2 = Dense(text_output_dim, activation='linear', name='dem_perc_scale')(dem_perc)
    bill_dem = Multiply(name="dem_scaled")([dem_perc2, bill_embed_dem])

    ### Add two copies of the bill 
    all_bill = Add()([bill_dem, bill_rep])

    # Shrink the combined bill down to legislator dimensions
    bill_scaled = Dense(k_leg_emb, activation='linear', name='bill_scaled')(all_bill)
    bill_scaled = Dropout(dropout_bill)(bill_scaled)

    # Take the product with the legislator embeddings
    bill_leg_product = Multiply(name='meta_leg_concat')([bill_scaled, leg_embed]) 
    bill_leg_product = Dropout(0.2)(bill_leg_product)

    main_output = Dense(1, activation="sigmoid", name="main_output",
                    )(bill_leg_product)

    model = Model(inputs=[leg_input, bill_input_rep, bill_input_dem, rep_perc, dem_perc], outputs=[main_output])

    model.compile(loss='binary_crossentropy',
              optimizer='adamax', metrics=['accuracy'])

    print(model.summary())

    return model

def prepare_two_party_model_features(vote_df, **kwargs):
    '''
    Create features necessary to run a metadata voting model. Takes in a vote data 
    pandas data frame and returns: (input features, predicted_votes), vectorizer, leg_crosswalk
    where vectorizer and leg_crosswalk are the id_maps (words -- ids) + (leg natural_ids -- ids)

    vote_df: Pandas data frame 
    kwargs: any settings necessary in the prepare_basic_model_features
    '''	
    # Prepare text and leg mappings through basic function
    all_basic_data, vectorizer, leg_crosswalk = prepare_basic_model_features(vote_df, **kwargs)
    
    # Set-up common system to map both train and test data to 
    # Only create a new one, if it has not already been defined.
    if leg_crosswalk is None:
        leg_ids = vote_df["leg_id"].unique()
        leg_crosswalk = pd.Series(leg_ids).to_dict()

    leg_crosswalk_rev = dict((v, k) for k, v in leg_crosswalk.items())
    vote_df['leg_id_v2'] = vote_df.leg_id.map(leg_crosswalk_rev)

    # If some legislators did not get a leg_id, drop them
    vote_df = vote_df[vote_df.leg_id_v2.notnull()]

    # Create sponsor party features
    meta_rep, meta_dem = prepare_sponsor_party_percs(vote_df)

    # Duplicate text as used in two places (for dem and rep bill)
    all_data = [all_basic_data[0], all_basic_data[1], all_basic_data[1], meta_rep, meta_dem]
    votes = vote_df["vote"].astype(int).values

    return (all_data, votes), vectorizer, leg_crosswalk

def evaluate_model(vote_df_train, vote_df_test, experiment_name='some_exp',
                    bill_type='basic', **kwargs):

    # Prepare all texts for evaluation
    all_train_data, vectorizer, leg_crosswalk = prepare_two_party_model_features(vote_df_train, **kwargs)
    all_test_data, _, _ = prepare_two_party_model_features(vote_df_test, vectorizer=vectorizer, leg_crosswalk=leg_crosswalk, **kwargs)
    all_2013, _, _ = prepare_two_party_model_features(vote_df_2013, vectorizer=vectorizer, leg_crosswalk=leg_crosswalk, **kwargs)
    all_2015, _, _ = prepare_two_party_model_features(vote_df_2015, vectorizer=vectorizer, leg_crosswalk=leg_crosswalk, **kwargs)

    word_embeddings = parse_glove_embeddings(vectorizer.vocabulary_)

    # Get distinct legs in train data
    n_leg = len(vote_df_train.leg_id_v2.unique())
    vocab_size = word_embeddings.shape[0]

    # Set-up model 
    callbacks = get_default_callbacks(experiment_name=experiment_name, use_checkpoint=True)

    model = get_two_party_meta_model(n_leg, vocab_size, word_embeddings=word_embeddings, 
                bill_type=bill_type, **kwargs)
    
    # Run model and evaluate
    model.fit_generator(batch_generator(*all_train_data),
          validation_data=all_test_data,
          steps_per_epoch=len(vote_df_train) / 50,
          epochs=50,
          verbose=1,
          class_weight={0: sample_weights[0],
                                  1: sample_weights[1]},
          callbacks=callbacks)
    
    print(model.evaluate(*all_2013))
    print(model.evaluate(*all_2015))
    
    return 100.


if __name__ == '__main__':

    prefix = "../data/us_data/"

    mode = 'summary' # 'fulltext'
    bill_type = 'cnn' # mwe
    experiment_name = 'meta_{}_{}_v1'.format(mode, bill_type)

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
        print(test_ids[0])

        train_bills = all_bill_ids[train_ids]

        print(train_bills[0])

        vote_df_train = vote_df_all[vote_df_all.natural_id.isin(train_bills)]
        vote_df_test = vote_df_all[~vote_df_all.natural_id.isin(train_bills)]

        score = evaluate_model(vote_df_train, vote_df_test, bill_type=bill_type,
                                n_word=n_word, k_leg_emb=k_leg_emb, 
                                experiment_name=experiment_name,text_field='special_text',
                                vocab_size=10000,
                                dropout_leg=dropout_leg, dropout_bill=dropout_bill,
                                dropout_bill2=dropout_bill2)

        print("SCORE:", score)

        all_scores.append(score)
        

    pickle.dump(all_scores, open('meta_model_scores_{mode}_{bill_type}.pkl', 'wb'))
