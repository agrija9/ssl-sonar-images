from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import argparse
from rotnet import RotNet

parser = argparse.ArgumentParser(description="Tensorflow 2 implementation of RotNet (ICLR 2018)")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--baseline_model', type=str, required=True)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--image_height", type=int, default=96)
parser.add_argument("--image_width", type=int, default=96)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--train", action='store_true')
parser.add_argument("--train_mode", type=str, required=True) # self_supervised_learning or supervised_learning
args = parser.parse_args()

def main():
    # Configure Tensorflow for GPU device
    tf.config.experimental_run_functions_eagerly(True)
    print("[INFO] Tensorflow Version:", tf.__version__)

    if tf.config.list_physical_devices("GPU") and tf.test.is_built_with_cuda():
        print("[INFO] Tensorflow built with CUDA")
        print("[INFO] Number GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        print("[INFO] List of GPU devices:", tf.config.list_physical_devices("GPU"))
        physical_devices = tf.config.list_physical_devices("GPU")
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)

    else:
        print("[ERROR] GPU not detected, make sure tensorflow-gpu is installed and that GPU is recognized")
        exit()

    # Define model
    print("[INFO] Creating RotNet model")
    rotation_net = RotNet(args)

    # Train model (pretraining)
    if args.train:
        print("[INFO] Started training model (pretraining)")
        rotation_net.train()

    # Run inference
    else:
        print("[INFO] Running inference w trained model")
        rotation_net.predict()


if __name__ == "__main__":
    main()
