import glob
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from util.dir import (CONFIG_FILENAME, CV_MODELS_DIR, DATA_PRED_DIR, DATA_VIS_DIR, MODELS_DIR, SEP, makedir)
from util.json import from_json
from util.preprocessing import pre_pad_protein_sequences
from util.reducer import tsne_2d, umap_2d
from util.tokenizer import load_tokenizer


COMBS = [
    ["bass_ntm_domain_test", "fass_ntm_domain_test", "fass_ctm_domain_test", "NLReff_test", "csga2203_nr40"],
    ["bass_c01_ntm_domain_test", "bass_c02_ntm_domain_test", "bass_c03_ntm_domain_test", "bass_c06_ntm_domain_test", "het-s_ntm_domain_test", "sigma_ntm_domain_test", "pp_ntm_domain_test", "csga2203ER4_nr40"]
]

def visualize(model_dir: str, layer_name: str) -> None:
    tokenizer = load_tokenizer()
    config = from_json(os.path.join(model_dir, CONFIG_FILENAME))

    for comb in COMBS:
        preds = []
        for set in comb:
            pred = pd.read_csv(os.path.join(model_dir, DATA_PRED_DIR, f'{set}.{config["modelcomb_name"]}.csv'), sep=SEP)
            pred["dataset"] = set
            preds.append(pred)
        preds = pd.concat(preds)
        frag = preds["frag"]

        # Pad protein sequences
        frag = pre_pad_protein_sequences(frag, config["T"])

        # Tokenize text
        frag = np.asarray(tokenizer.texts_to_sequences(frag))

        # Collect multidimensional representations from cv models
        mdim_rep = []
        for model_filepath in glob.glob(os.path.join(model_dir, CV_MODELS_DIR, "*")):
            # Load model
            model = tf.keras.models.load_model(model_filepath)

            # Get output from selected layer
            layer_out = model.get_layer(layer_name).output
            fun = tf.keras.backend.function(model.input, layer_out)
            mdim_rep.append(fun(frag))

        # Dimension reducers
        mdim_rep = np.concatenate(mdim_rep, axis=1)
        preds["tsne_x"], preds["tsne_y"] = tsne_2d(mdim_rep)
        preds["umap_x"], preds["umap_y"] = umap_2d(mdim_rep, n_neighbors=5)

        # Save results
        preds.to_csv(makedir(os.path.join(model_dir, DATA_VIS_DIR, f'{".".join(comb)}.csv')), sep=SEP, index=False)
        

if __name__ == "__main__":
    model_dir = os.path.join(MODELS_DIR, sys.argv[1])
    layer_name = "before-classif" if len(sys.argv) < 3 else sys.argv[2] 
    visualize(model_dir, layer_name)
