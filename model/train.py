import time
from utils import *
from model.PocketWriterModel import *
from model.OneStepWriterModel import *


def train_model(dataset_folder, epochs, output_dir="data/models", charDataSet=True):
    
    corpus = load_corpus(dataset_folder)
    vocab = create_vocab(corpus, charVocab=charDataSet)
    
    vocab_ids = translate_input_for_the_rnn(vocab)
    chars_from_ids = translate_rnn_output(vocab_ids)

    dataset = create_dataset(corpus, vocab_ids, charDataset=charDataSet)
    
    model = PocketWriterModel(
        vocab_size=len(vocab_ids.get_vocabulary()),
        embedding_dim=256,
        rnn_units=1024,
    )
    
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss)
    checkpoint_dir = "./data/checkpoints"
    
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True
    )

    #train
    model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
    one_step_model = OneStepWriterModel(model, chars_from_ids, vocab_ids)
    
    #testing prediction
    start = time.time()
    states = None
    next_char = tf.constant(["Aureliano:"])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)
    
    if charDataSet:
        result = tf.strings.join(result)
    else:
        result = tf.strings.join(result, separator=" ")
    end = time.time()
    print(result[0].numpy().decode("utf-8"), "\n\n" + "_" * 80)
    print("\nRun time:", end - start)

    #saving
    tf.saved_model.save( one_step_model, output_dir)