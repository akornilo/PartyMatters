# PartyMatters
Data and code for [Party Matters: Enhancing Legislative Embeddings with Author Attributes for Vote Prediction](https://arxiv.org/abs/1805.08182)

The dataset can be found [here](https://drive.google.com/drive/folders/1NIV9ieyHab67UjDrpGwKtyJXo2-hzh_l?usp=sharing). The file DataDescription.md explains the format.


# Process Description

The dependencies are specified in the `enviornment.lst` file - it was created with miniconda, but a pip install should work.

## Cleaning the data


## Running the different models

To run all the models - go into the `models` folder and set `PYTHONPATH=.`

### Text-only models

The MWE, MWE+FT and CNN models from the paper can be run from the `models/basic_layers.py` file. 

To switch between the models, change the `mode` and `bill_type` variables at the top of the `if __name__ == '__main__'` code.


### Text + Meta Models

The MWE+Meta, MWE+FT+Meta, CNN+Meta models can be run from the `models/meta_models.py` file - its setup the same at the basic_layers file.

To switch between the models, change the `mode` and `bill_type` variables at the top of the `if __name__ == '__main__'` code.


### Meta-only

TBD -- need to fix some details here