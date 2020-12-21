## BSLDict Dataset Documentation

The contents of `bsldict_v1.pkl` file is structured as follows:

* Video naming convention: `<letter>_<03d:page>_<03d:sign_cnt>_<03d:video_cnt_in_sign>_<sign_text>.mp4`
* Note: due to 88 videos for which i3d/mlp features are missing, we release 14122 videos, corresponding to 9261 words (instead of 14210 and 9283).

``` bash
{
    ["words"]                   # 9283 word vocabulary (has "-" character for multi-word phrases)
    ["words_to_id"]             # dictionary from word => a unique word id 
    ["words_normalised"]        # normalised text (e.g., "1" becomes "one"), this removes "-" and "?" characters from the original word and runs normalise from https://pypi.org/project/normalise/
    # Dictionary containing metadata for each of the 14K videos
    # each key represents a list of 14K values
    ["videos"]{
        name                    # our unique naming, filename of the .mp4 video
        word                    # sign_text_orig_db if exists (signers seem to mouth this word, otherwise sign_text_db)
        word_id                 # word number corresponding to the 'word'
        alternative_words       # a list of other words for which the same video url appears (mostly empty)
        # Items with *_db suffix are metadata obtained from signbsl.com
        letter_db               # initial letter of the sign
        page_db                 # page number of the sign
        sign_cnt_db             # index of the sign in one page
        sign_text_orig_db       # sign text from the original website (might be missing)
        sign_text_db            # sign
        sign_link_db            # relative link to the sign's website, e.g., /sign/eye
        # Items common to all videos within that version
        version_cnt_db          # number of versions for this sign (one sign might have multiple versions)
        how_to_db               # textual description of the sign or the meaning, sometimes with usage examples in quotes
        see_also_db             # list of signs that are related
        similar_items_db        # list of signs that are similar/same
        categories_db           # list of signs
        within_this_cat_db      # list of signs within the same category
        # Items specific to that video
        video_cnt_in_version_db # index of the video within that version
        video_cnt_in_sign_db    # index of the video within that sign
        video_link_db           # link to the video file (either .mp4 file or youtube link)
        source_site_db          # Signstation, Scottish Sensory Centre, QIA Dictionary of ICT,  Nathanael Farley, University of Wolverhampton, Karl O'Keeffe ... (might be missing)
        download_method_db      # wget | youtube-dl
        youtube_identifier_db   # youtube id if a youtube link, None otherwise
        upload_date_db          # upload date (might be missing)
        # The following two items are from our extractions using OpenPose
        bbox                    # normalised [0, 1] bounding box coordinates [y0, x0, y1, x1]
        temp_interval           # the beginning and end frame indices (inclusive) that have high motion (higher than 5 pixels distance) for wrists
        # Dictionary containing our I3D and MLP feature extractions
        ["features"]{
            i3d                 # 1024-dimensional I3D features (randomly sampled 16f clip features are averaged)
            mlp                 # 256-dimensional MLP features (used for sign spotting)
        }
        # Dictionary containing video resolution information for the original videos
        ["videos_original"]{
            T                   # number of frames
            H                   # height
            W                   # width
            duration_sec        # seconds
            fps                 # frames per second
        }
        # Dictionary containing video resolution information for our preprocessed version
        # Resized such that the height is 360 pixels, resampled at 25 fps
        ["videos_360h_25fps"]{
            T                   # number of frames
            H                   # height, 360
            W                   # width
            duration_sec        # seconds
            fps                 # frames per second
        }
    }
}
```
