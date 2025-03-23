# Maximum caption length (including <SOS>)
TEXT_MAX_LEN = 150+1

#Tokenizer modes
CHAR_LEVEL = "char"
WORDPIECE_LEVEL = "wordpiece"
WORD_LEVEL = "word"

#Instantiate tokenizer
#Options: CHAR_LEVEL, WORDPIECE_LEVEL, WORD_LEVEL
#TOKENIZATION_MODE = CHAR_LEVEL
#TOKENIZATION_MODE = WORDPIECE_LEVEL
TOKENIZATION_MODE = WORD_LEVEL

#Directories
ROOT_DIR = "/ghome/c5mcv06/FIRD"
OUTPUT_DIR = f"./res_finals/{TOKENIZATION_MODE}_prova_model2"

#Training config
BATCH_SIZE = 64
NUM_WORKERS = 4