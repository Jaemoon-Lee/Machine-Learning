################################################################################
#  Generate melspectrogram training set from all prog and nonprog songs in dir #
#  Each pattern is a 20 (Dim of a sample) by NUM_OF_SAMPLE_PER_WINDOW matrix.  #
#  (This file generates two .pkl file, prog and nonprog respectively)          #
################################################################################

from featureExtractTool import generate_patterns_for_all_songs_in_dir

# Define pattern size
NUM_OF_SAMPLE_PER_WINDOW = 43
NUM_OF_OVERLAP_SAMPLE = 0

# Set prog songs
NAME_OF_PROG_PATTERNS = "progPatterns_43_offset30_.pkl"
PATH_TO_PROG_SONG_DIR = "../training songs/prog"
PROG_LABEL = "prog"

# Set nonprog songs
NAME_OF_NONPROG_PATTERNS = "nonprogPatterns_43_offset30_.pkl"
PATH_TO_NONPROG_SONG_DIR = "../training songs/nonprog"
NONPROG_LABEL = "nonprog"


# Get patterns for prog songs
generate_patterns_for_all_songs_in_dir( PATH_TO_PROG_SONG_DIR,
                                        PROG_LABEL,
                                        NUM_OF_SAMPLE_PER_WINDOW,
                                        NUM_OF_OVERLAP_SAMPLE,
                                        NAME_OF_PROG_PATTERNS )

"""
# Get patterns for nonprog songs
generate_patterns_for_all_songs_in_dir( PATH_TO_NONPROG_SONG_DIR,
                                        NONPROG_LABEL,
                                        NUM_OF_SAMPLE_PER_WINDOW,
                                        NUM_OF_OVERLAP_SAMPLE,
                                        NAME_OF_NONPROG_PATTERNS )
"""
