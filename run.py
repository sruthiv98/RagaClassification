#!/usr/bin/env python

import sys
import json
import shutil

sys.path.insert(0, 'src') # add library code to path
from etl import load, clean_data 
from model import driver
from features import make_features

DATA_PARAMS = 'config/01-data.json'
CLEAN_PARAMS = 'config/02-clean.json'
FEATURE_PARAMS = 'config/03-features.json'
MODEL_PARAMS = 'config/04-model.json'

TEST_DATA_PARAMS = 'config/test-01-data.json'
TEST_CLEAN_PARAMS = 'config/test-02-clean.json'
TEST_FEATURE_PARAMS = 'config/test-03-features.json'
TEST_MODEL_PARAMS = 'config/test-04-model.json'


def load_params(fp):
    '''
    Loads parameters.
    '''
    with open(fp) as fh:
        param = json.load(fh)

    return param


def main(targets):
    '''
    Reads targets and executes appropriate files for given data. 
    '''

    # make the clean target
    if 'clean' in targets:
        shutil.rmtree('data/raw', ignore_errors=True)
        shutil.rmtree('data/cleaned', ignore_errors=True)
        shutil.rmtree('data/out', ignore_errors=True)
        shutil.rmtree('data/test', ignore_errors=True)

    # make the data target
    if 'data' in targets:
        cfg = load_params(DATA_PARAMS)
        load(**cfg)

        cfg = load_params(CLEAN_PARAMS)
        clean_data(**cfg)

    # make the test target
    if 'test-project' in targets:
        cfg = load_params(TEST_DATA_PARAMS)
        load(**cfg)

        cfg = load_params(TEST_CLEAN_PARAMS)
        clean_data(**cfg)

        cfg = load_params(TEST_FEATURE_PARAMS)
        make_features(**cfg)

        cfg = load_params(TEST_MODEL_PARAMS)
        driver(**cfg)

    # make the full data target
    if 'full-project' in targets:
        cfg = load_params(DATA_PARAMS)
        load(**cfg)

        cfg = load_params(CLEAN_PARAMS)
        clean_data(**cfg)

        cfg = load_params(FEATURE_PARAMS)
        make_features(**cfg)

        cfg = load_params(MODEL_PARAMS)
        driver(**cfg)


    # if data is cleaned and just model pipeline is to be run
    if 'model' in targets:
        cfg = load_params(TEST_MODEL_PARAMS)
        driver(**cfg)

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
