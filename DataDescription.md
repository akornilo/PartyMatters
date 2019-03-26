# Data Table Description

There are three datasets associated with the paper Party Matters: Enhancing Legislative Embeddings with Author Attributes for Vote Prediction.

Each dataset is stored as a python pandas dataframe.

## Voting Data

full_data_20052012.pkl - Voting data from 2005-2012 - this was used for training and "in-session" testing
full_data_20132014.pkl - Data for 2013-2014 session (114th) - used for evaluation
full_data_20152015.pkl - Data for 2015-2016 session (115th) - used for evaluation

The data table contains one row per vote.

Column definitions:

#### Bill Features
- natural_id : unique bill identifier (Session + Bill Number)
- bill_type: type of legislation (bill, resolution, joint_resolution, concurrent_resolution)
- chamber: chamber in which bill originated (upper or lower)
- pip: current party in power in the chamber (d or r)

- d_perc/r_perc/i_perc : percent democrat/republican/independent sponsors respectively
- total_sponsors: number of bill sponsors

- title: Bill title
- summary: Bill summary

#### Voting Legislator features 

- leg_id: unique legislator identified
- leg_name: legislator's name
- party: legislator's party
- vote: 0 if voted "no", 1 if "yes"

##  Bill Text Data

The bill texts are stored in a separate file with one row per bill: bill_text_[20052012, 20132014, 20152016].pkl


The columns are:
natural_id: bill identifier that matches the one above
text: the raw text of the bill
clean_text: stemmed and lowercased version of the text

Note: Bills will contain multiple text versions, in this paper we will always use the latest version.



