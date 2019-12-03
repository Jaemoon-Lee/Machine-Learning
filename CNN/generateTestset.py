################################################################################
#  Generate melspectrogram ""TEST"" set from all prog and nonprog songs in dir #
#  Each pattern is a 20 (Dim of a sample) by NUM_OF_SAMPLE_PER_WINDOW matrix.  #
#  (This file generates two .pkl file, prog and nonprog respectively)          #
################################################################################

from featureExtractTool import generate_test_set

# Define pattern size
NUM_OF_SAMPLE_PER_WINDOW = 43
NUM_OF_OVERLAP_SAMPLE = 0

# Set prog songs
NAME_OF_PROG_PATTERNS = "progPatterns_TESTSET_43_offset30.pkl"
PATH_TO_PROG_SONG_DIR = "../validation songs/Prog"
PROG_LABEL = "prog"

# Set nonprog songs
NAME_OF_NONPROG_PATTERNS = "nonprogPatterns_TESTSET_43_offset30.pkl"
PATH_TO_NONPROG_SONG_DIR = "../validation songs/Non-Prog"
NONPROG_LABEL = "nonprog"

"""
# Get patterns for prog songs
generate_test_set( PATH_TO_PROG_SONG_DIR,
                   PROG_LABEL,
                   NUM_OF_SAMPLE_PER_WINDOW,
                   NUM_OF_OVERLAP_SAMPLE,
                   NAME_OF_PROG_PATTERNS )


"""
# Get patterns for nonprog songs
generate_test_set( PATH_TO_NONPROG_SONG_DIR,
                   NONPROG_LABEL,
                   NUM_OF_SAMPLE_PER_WINDOW,
                   NUM_OF_OVERLAP_SAMPLE,
                   NAME_OF_NONPROG_PATTERNS )
