""" This script prepares the data-manifest files (in JSON format)
for training and testing a Voice Activity Detection system with the
LibriParty dataset.

The dataset contains sequences of 1-minutes of LibiSpeech sentences
corrupted by noise and reverberation. The dataset can be downloaded
from here:

https://drive.google.com/file/d/1--cAS5ePojMwNY5fewioXAv9YlYAWzIJ/view?usp=sharing

Authors
 * Mohamed Kleit 2021
  * Arjun V 2021
"""

# import numpy as np
# import pandas as pd
# import json
import logging
import torchaudio

# from collections import OrderedDict

import deeplake
from torch import from_numpy

# import json

from libriparty_prepare import create_window_splits
from libriparty_prepare import remove_duplicates_sort
from libriparty_prepare import save_dataset


""" Global variables"""
logger = logging.getLogger(__name__)
valid_json_dataset = {}


def add_example_ubenwa(rec_id, cry, window, example, sample_rate, json_dataset):
    example = "example_" + str(example)
    json_dataset[example] = {}
    json_dataset[example]["record"] = {}
    json_dataset[example]["record"]["rec_id"] = rec_id
    json_dataset[example]["record"]["start"] = window[0] * sample_rate
    json_dataset[example]["record"]["stop"] = window[1] * sample_rate
    for interval in cry:
        interval[0] -= window[0]
        interval[1] -= window[0]
    json_dataset[example]["cry"] = cry
    return json_dataset


def create_json_dataset_ubenwa(dic, sample_rate, window_size):
    """Creates JSON file for Voice Activity Detection.
    Data are chunked in shorter clips of duration window_size"""

    recid_iteration = dic.keys()
    example_counter = 1
    json_dataset = {}

    for record in recid_iteration:
        try:

            cry_timings = dic[record]["cry"]
            reference_list = []
            compare_list = []
            for values in cry_timings:
                reference_list, compare_list = create_window_splits(
                    values, compare_list, reference_list, window_size
                )
                unique_list, seq_timing = remove_duplicates_sort(
                    reference_list, compare_list
                )
            speech_sequence_cleaned = []
            overlap = []
            for i, values in enumerate(unique_list):
                if len(values) == 1:
                    speech_sequence_cleaned.append(seq_timing[values[0]])
                    json_dataset = json_dataset = add_example_ubenwa(
                        dic[record]["rec_id"],
                        [seq_timing[values[0]]],
                        compare_list[values[0]],
                        example_counter,
                        sample_rate,
                        json_dataset,
                    )
                    example_counter += 1
                else:
                    for iter in values:
                        overlap.append(seq_timing[iter])
                    json_dataset = add_example_ubenwa(
                        dic[record]["rec_id"],
                        overlap,
                        compare_list[values[0]],
                        example_counter,
                        sample_rate,
                        json_dataset,
                    )
                    speech_sequence_cleaned.append(overlap)
                    overlap = []
                    example_counter += 1
                dic[record]["cry_segments"] = speech_sequence_cleaned
        except Exception as e:
            logging.error(e)
            logging.info(f"Record {record} has no cry: skipped.")
            continue

    return json_dataset


def create_json_structure_ubenwa(ds):
    json = {}
    for i, sample in enumerate(ds):
        list_cry = []
        rec_id = sample.record_id.data()["value"]
        segments_E = (
            sample.segments_E.numpy(fetch_chunks=True) / 16000
        ).tolist()
        segments_EN = (
            sample.segments_EN.numpy(fetch_chunks=True) / 16000
        ).tolist()

        list_segments_E = [tuple(sub) for sub in segments_E]
        list_segments_EN = [tuple(sub) for sub in segments_EN]

        list_cry = list_segments_E + list_segments_EN

        json["session_" + str(rec_id)] = {"rec_id": rec_id, "cry": list_cry}
    return json


def read_audio_ubenwa(ds: deeplake.Dataset, record):
    """General audio loading for Ubenwa data from deeplake dataset.

    Expected use case is in conjunction with Datasets
    specified by JSON.


    Arguments
    ----------
    ds : deeplake.Dataset
        deeplake dataset or a view form the dataset,
    rec_id : str,
        Record id of the sample to get the audio data from deeplake


    Returns
    -------
    torch.Tensor
        1-channel: audio tensor with shape: `(samples, )`


    """
    # filter based on the rec_id
    list_of_rec_ids = ds[:].record_id.data()["value"]
    rec_index = list_of_rec_ids.index(str(record["rec_id"]))
    sample = ds[rec_index]

    # rec_dict = {"Index": [], "Rec_id": []}
    # sample = ds.filter(lambda sample: sample.record_id.data()['value'] == record["rec_id"], progressbar = False)
    raw_audio = sample["raw_audio"].numpy().flatten()
    audio = from_numpy(raw_audio).float() / 32768.0

    return audio


def prepare_ubenwa(
    dataset_path: str = "/Users/sajjadabdoli/Documents/Ubenwa/data/ub-processed/seg-191101-230303/",
    save_json_folder: str = None,
    sample_rate: int = 16000,
    window_size: int = 5,
    skip_prep: bool = False,
):

    """
    Prepares the Json files for the train, validation and the test sets.
    dataset_path: str
        The path to the location of the deeplake dataset
    save_json_folder:
        The path where to store the valid json file.
        The path where to store the valid json file.
    sample_rate:
    window_size:
        Size of each audio frame
    skip_prep:
        Default: False
        If True, the data preparation is skipped.
    """

    ds = deeplake.load(
        dataset_path + "deeplake/",
        read_only=True,
        memory_cache_size=8192,
        local_cache_size=20480,
    )

    # load the views from the deeplake dataset

    ds_train = ds.load_view(id="train")
    ds_dev = ds.load_view(id="dev")
    ds_test = ds.load_view(id="test")

    # Create json structure
    train_dict = create_json_structure_ubenwa(ds_train)
    valid_dict = create_json_structure_ubenwa(ds_dev)
    test_dict = create_json_structure_ubenwa(ds_test)

    # Create datasets as json
    train_dataset = create_json_dataset_ubenwa(
        train_dict, sample_rate, window_size
    )
    valid_dataset = create_json_dataset_ubenwa(
        valid_dict, sample_rate, window_size
    )
    test_dataset = create_json_dataset_ubenwa(
        test_dict, sample_rate, window_size
    )

    # Save datasets
    save_dataset(save_json_folder + "/train.json", train_dataset)
    save_dataset(save_json_folder + "/valid.json", valid_dataset)
    save_dataset(save_json_folder + "/test.json", test_dataset)
