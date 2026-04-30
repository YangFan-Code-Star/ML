import os


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DATA_OUT = os.path.join(PROJECT_ROOT, "output")
TEST_TARGET_PATH = os.path.join(DATA_OUT, "test_target.csv")
PLOTS_DIR = os.path.join(DATA_OUT, "plots")

EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 0
LEARNING_RATE = 0.01
MOMENTUM = 0.5

HIDDEN_DIM1 = 128
HIDDEN_DIM2 = 64

VALID_RATIO = 0.2
RANDOM_SEED = 42
