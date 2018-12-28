import sys

# add path for transformer
PROJECT_PATH = '..'
TRANSFORMER_PATH = '../transformerpy'
NTM_PATH = '../pytorch-ntm'

if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
if TRANSFORMER_PATH not in sys.path:
    sys.path.append(TRANSFORMER_PATH)
if NTM_PATH not in sys.path:
    sys.path.append(NTM_PATH)

WORD_VECTORS = '../embeddings/crawl-300d-2M.vec'
# WORD_VECTORS = '../embeddings/glove.840B.300d.txt'
EMBEDDING_DIM = 300