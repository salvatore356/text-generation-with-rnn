import time
from utils import *

def predict(model_folder, initial_word="Aureliano:", isWordLevel=False):
    print("Loading model...")
    one_step_reloaded = tf.saved_model.load(model_folder)

    word = input("Say something to your model (exit to close): ")
    
    if not word:
        word = initial_word
    output_lenght = 1000
    if isWordLevel:
        output_lenght = 100
    while word != "exit":
        start = time.time()
        states = None
        next_char = tf.constant([word])
        result = [next_char]

        for n in range(output_lenght):
            next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
            result.append(next_char)

        if isWordLevel:
            result = tf.strings.join(result, separator=" ")
        else:
            result = tf.strings.join(result)
        end = time.time()
        print(result[0].numpy().decode("utf-8"), "\n\n" + "_" * 80)
        print("\nRun time:", end - start)

        word = input("Say something to your model (exit to close): ")