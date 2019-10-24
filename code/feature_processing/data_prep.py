'''
Just setup dataset
'''
import pandas as pd 

from utilities.resro_helper import ResroHelper
from utilities.text_preprocessing import TextPreprocessor

data = pd.read_pickle('../data/us_data/train_20052012_all.pkl')
data2 = pd.read_pickle('../data/us_data/test_20052012_all.pkl')
data = pd.concat([data, data2])

# First fetch all titles and summaries

resro = ResroHelper()

bills = data.groupby('natural_id').first()

new_titles = {}
new_summaries = {}
for bid in bills.index:

    locality, session, eid = bid.split('_')

    query = ''' SELECT title, summary 
                FROM legislation.bills 
                WHERE locality='us' and session=:session and external_id=:eid
            '''

    title, summary = resro.session.execute(query, params={'session': session, 'eid': eid}).fetchone()    

    new_titles[bid] = title 
    new_summaries[bid] = summary

data['title'] = data.natural_id.map(new_titles)
data['summary'] = data.natural_id.map(new_summaries)


# Next let's clean them up
tp = TextPreprocessor()

# Summaries first
summaries = data.groupby('natural_id').first().summary
text_map = dict(summaries)
text_map2 = {}
for k,v in text_map.items():
    v = v or '' # Sometimes summaries are null
    text_map2[k] = ' '.join(tp.tokenize(v))

data['summary_clean_tp'] = data.natural_id.map(text_map2)

# Clean up the title too
text_fields = data.groupby('natural_id').first().title
text_map = dict(data.groupby('natural_id').first().title
)
text_map2 = {}

for k,v in text_map.items():
    text_map2[k] = ' '.join(tp.tokenize(v))

data['title_clean_tp'] = data.natural_id.map(text_map2)

data['title_summary_clean_tp'] = data['title_clean_tp'] + ' ' + data['summary_clean_tp']

# For our final field - use title+summary. If too short, use the text 

final_fields = {}
for i, row in data.groupby('natural_id').first().iterrows():
    
    title_summary = row.title_clean_tp + row.summary_clean_tp
    
    if len(set(title_summary.split())) < 50:
        # Clean up text first
        text = ' '.join(tp.tokenize(row.text))
        final_fields[i] = text
    else:
        final_fields[i] = title_summary
data['special_text'] = data.natural_id.map(final_fields)

# For space purposes - only save key fields
data = data[['natural_id', 'chamber', 'leg_id', 'd_perc', 'r_perc', 'special_text', 'party', 'vote']]

data.to_pickle('../data/us_data/full_20052012_all_summary.pkl')
