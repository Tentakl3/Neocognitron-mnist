import trainer
import pickle
import os

filename = 'data/network/neocognitron_trained'

def save_object(obj, file):
    with open(file, "wb") as file:
        pickle.dump(obj, file)

def load_object(file):
    if os.path.exists(file):  # Check if the file exists
        with open(file, "rb") as file:
            return pickle.load(file)
    return None  # Return None if the file does not exist

if os.path.exists(filename):
    network = load_object(filename)
    trainer.validate(network)
else:
    network = trainer.runTraining()
    save_object(network, filename)