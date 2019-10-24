'''
Modify the dataset to use clean texts
'''
import os
import pandas as pd 

from feature_processing.text_preprocessor import TextPreprocessor

# Change prefix to point to your data directory
# Make a clean_text directory
PREFIX = '../data/raw_data'
CLEAN_PREFIX = '../data/clean_data'

def clean_summary(session):

    data = pd.read_pickle(os.path.join(PREFIX, f'full_data_{session}.pkl'))
    # Fix nulls in some summaries
    data = data.fillna('')

    text_data = pd.read_pickle(os.path.join(PREFIX, f'bill_text_{session}.pkl'))

    tp = TextPreprocessor()

    final_texts = {}

    for _, row in text_data.iterrows():

        clean_text = ' '.join(tp.tokenize(row.full_text))

        final_texts[row.natural_id] = clean_text


    final_summary = {}

    for i, row in data.groupby('natural_id').first().iterrows():

        title_summary = ' '.join([row.title, row.summary])

        tokens = list(tp.tokenize(title_summary))
        clean_text = ' '.join(tokens)

        # If the resulting text is less than 20 words - attach the bill text
        if len(tokens) < 20:
            clean_text += ' ' + final_texts[i]

        final_summary[i] = clean_text

    data['target_summary'] = data.natural_id.map(final_summary)

    data['target_text'] = data.natural_id.map(final_texts)

    data.to_pickle(os.path.join(CLEAN_PREFIX, f'full_data_{session}_clean.pkl'))


if __name__ == '__main__':

    for session in ['20052012', '20132014', '20152016']:
        clean_summary(session)
