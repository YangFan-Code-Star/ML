import os


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 0
LEARNING_RATE = 0.01
VALID_RATIO = 0.1
RANDOM_SEED = 42
