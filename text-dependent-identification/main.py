import tensorflow as tf
import os
from model import train, test
from configuration import get_config
from data_preprocess import preprocess_test_tdsv, preprocess_train_tdsv

config = get_config()
tf.reset_default_graph()

if __name__ == "__main__":
    # start training
    if config.train:
        preprocess_train_tdsv()
        print("\nTraining Session")
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
        train(config.model_path)
    # start test
    else:
        preprocess_test_tdsv()
        print("\nTest session")
        if os.path.isdir(config.model_path):
            test(config.model_path)
        else:
            raise AssertionError("model path doesn't exist!")