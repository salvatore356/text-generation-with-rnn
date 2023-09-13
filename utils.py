import os
import re
import tensorflow as tf

def get_all_files_in_folder(folder):
    """
    Get a list of file paths within a folder.
    
    Args:
        folder_path (str): The path to the folder.
        
    Returns:
        List[str]: A list of file paths.
    """
    return [os.path.join(folder, f) for f in os.listdir(folder)]

def load_corpus(folder):
    """
    Load and concatenate text data from multiple files in a folder.

    Args:
        folder_path (str): The path to the folder containing text files.

    Returns:
        str: The concatenated text corpus.
    """
    paths_to_file = get_all_files_in_folder(folder)
    
    corpus = ""
    # Load text data from each file and concatenate
    for path_to_file in paths_to_file:
        corpus +=" " + open(path_to_file, "rb").read().decode(encoding="utf-8")
    return corpus

def create_vocab(corpus, charVocab=True):
    """
    Create a vocabulary from a corpus.

    Args:
        corpus (str): The text corpus.

    Returns:
        dict: A dictionary mapping words to word IDs.
    """
    if charVocab:
        return sorted(set(corpus))
    else:
        vocab = set()
        
        words = re.split(r'[ \n,.\t]+', corpus) #corpus.split(' ')  # Split the text into words
        
        vocab.update(words)
        
        #print(sorted(vocab))
        #file = open("vocab.txt", "w")
        #for word in sorted(vocab):
        #    file.write(word + "\n")

        #file.close()

        return sorted(vocab)


def translate_input_for_the_rnn(vocab):
    """
    Translate a vocabulary for the RNN.
    """
    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None
    )
    return ids_from_chars

def translate_rnn_output(ids):
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids.get_vocabulary(), invert=True, mask_token=None
    )
    return chars_from_ids

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def create_dataset(corpus, vocab_ids, sequence_length=100, batch_size=64, buffer_size=10000, charDataset = True):
    # Translate text to numerical representations
    if charDataset:
        all_ids = vocab_ids(tf.strings.unicode_split(corpus, "UTF-8"))
    else: 
        all_ids = vocab_ids(tf.strings.split(corpus))
    
    # Create a dataset from numerical ids
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    
    # Create sequences of fixed length
    sequences = ids_dataset.batch(sequence_length + 1, drop_remainder=True)
    
    # Split sequences into input and target
    dataset = sequences.map(split_input_target)
    
    # Shuffle, batch, and prefetch the dataset for better performance
    dataset = (
        dataset
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return dataset








'''
corpus = load_corpus("data/training/es")
vocab = create_vocab(corpus)
ids = translate_input_for_the_rnn(vocab)
translation = translate_rnn_output(ids)
dataset = create_dataset(corpus)

print(f"corpus length: {len(corpus)}")
print(f"vocab length: {len(vocab)}")
print(f"ids: {ids}")
print(f"translation: {translation}")
'''

