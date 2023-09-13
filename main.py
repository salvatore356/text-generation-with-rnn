from model.train import *
from model.predict import *

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    print("\nPocketWriter: Imperfectly Perfect Prose")
    choice = 0
    while choice != 3:
        print("\nOptions:")
        print("1. Train")
        print("2. Predict")
        print("3. Close")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            dataset_path = input("Enter the path to the dataset: (e.g. data/training/es) ")
            if not dataset_path:
                dataset_path = 'data/training/es'


            while True:
                level_input = input("Character level (1) or Word level (2): ")

                try:
                    level = int(level_input) if level_input else 10
                    if level > 2 or level < 1:
                        raise ValueError
                    break  # Exit the loop if a valid input is received
                except ValueError:
                    print("Invalid input. Please enter 1 or 2 for training level.")
            
            while True:
                epochs_input = input("Enter the epochs (e.g. 10): ")

                try:
                    epochs = int(epochs_input) if epochs_input else 10
                    break  # Exit the loop if a valid input is received
                except ValueError:
                    print("Invalid input. Please enter a valid integer for epochs.")
            
            print(f"Training for {epochs} epochs...")
            charDataSet = True
            if level == 1:
                charDataSet = False
            train_model(dataset_path, int(epochs), output_dir="data/models/OneStepWriterModelWordLevel", charDataSet = charDataSet)

        elif choice == '2':
            model_path = input("Enter the path to your model: (e.g. data/models/OneStepWriterModelWordLevel) ")
            if not model_path:
                model_path = 'data/models/OneStepWriterModelWordLevel'
            
            while True:
                level_input = input("Character level (1) or Word level (2): ")

                try:
                    level = int(level_input) if level_input else 10
                    if level > 2 or level < 1:
                        raise ValueError
                    break  # Exit the loop if a valid input is received
                except ValueError:
                    print("Invalid input. Please enter 1 or 2 for training level.")
            charDataSet = True
            if level == 1:
                charDataSet = False
            predict(model_path, isWordLevel=charDataSet)
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select a valid option (1/2/3).")

if __name__ == "__main__":
    main()