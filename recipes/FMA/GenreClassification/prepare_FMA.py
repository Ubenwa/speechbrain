import os
import logging

try:
    import pandas as pd
except ImportError:
    err_msg = (
        "The optional dependency pandas must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install pandas`.\n"
    raise ImportError(err_msg)

logger = logging.getLogger(__name__)

def prepare_FMA(
    tracks_metadata,
    audio_folder,
    save_folder,
    skip_prep=False,
):

    if skip_prep:
        return

    # All metadata fields to appear within our dataset annotation files (i.e. train.csv, valid.csv, test.cvs)
    fields = {
        "ID": [],
        "duration": [],
        "mp3": [],
        "genre": [],
    }
    tracks = pd.read_csv(tracks_metadata, index_col=0, header=[0, 1])
    logger.info(f"Loaded {len(tracks)} audio tracks")

    tracks_with_genre = tracks['track'].loc[tracks['track']['genre_top'].notnull()]
    tracks_with_genre = tracks_with_genre[['genre_top']]
    logger.info(f"Keep {len(tracks_with_genre)} tracks with genre")

    
    tracks_with_genre['audio_name'] = tracks_with_genre.index.map(str).str.zfill(6) + ".mp3"
    tracks_with_genre['audio_path'] = audio_folder + "/" + \
                                tracks_with_genre['audio_name'].str[0:3] + "/" + \
                                tracks_with_genre['audio_name']

    tracks_with_genre['duration'] = 30
    tracks_with_genre['ID'] = tracks_with_genre.index
    
    train_df = tracks_with_genre.loc[tracks['set', 'split'] == 'training']
    valid_df = tracks_with_genre.loc[tracks['set', 'split'] == 'validation']
    test_df = tracks_with_genre.loc[tracks['set', 'split'] == 'test']
        
    train_df.to_csv(os.path.join(save_folder, "train.csv"))
    valid_df.to_csv(os.path.join(save_folder, "valid.csv"))
    test_df.to_csv(os.path.join(save_folder, "test.csv"))


if __name__ == "__main__":
    prepare_FMA(
        "~/datasets/fma_metadata/tracks.csv",
        "~/datasets/fma_large/",
        "tmp"
    )
