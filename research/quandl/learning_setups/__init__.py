import time
from keras.callbacks import TensorBoard


def get_log_dir(setup_name):
    return "logs/{}-{}".format(time.strftime("%Y-%m-%dT%H-%M-%S"), setup_name)


def create_tensor_board(setup_name):
    return TensorBoard(
        log_dir=get_log_dir(setup_name),
        histogram_freq=10
    )
