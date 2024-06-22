import tensorflow as tf
import cv2
import os 
from tensorflow.keras import optimizers, losses

train_directory = "../data/training"
test_directory = "../data/test"
batch_size = 32
validation_split = 0.2
num_classes = 2


def get_data(directory, batch_size, validation_split, subset):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels ="inferred",
        batch_size=batch_size,
        label_mode ="int",
        validation_split = validation_split,
        subset=subset,
        seed=5,
        )

def create_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(64, 9, activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2), (2,2)),
        tf.keras.layers.Conv2D(32, 4, activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2), (2,2)),
        tf.keras.layers.Conv2D(16, 2, activation='relu'),
        tf.keras.layers.Conv2D(16, 2, activation='relu'),
        tf.keras.layers.Conv2D(16, 2, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    return model


def print_invaild_files(directory):
    vaild_filetypes = ["jpeg", "png"]
    for file in os.scandir(directory):
        file_ext = file.name.split(".")[1]
        if file_ext.lower() not in vaild_filetypes:
            os.remove(file.path)
        else:
            print("Found vaild file ", file.path)

def get_res_count(directory):
    dict_height = {}
    dict_width = {}
    for file in os.scandir(directory):
        current_picture = cv2.imread(file.path)
        shape = current_picture.shape
        h,w,_ = shape 
        dict_height[h] = dict_height.get(h,0)+1
        dict_width[w] = dict_width.get(w,0)+1
    return dict_height, dict_width


def main():

    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # print_invaild_files("../data/test/Bike")
    # print_invaild_files("../data/test/Car")
    # dH, dW = get_res_count(train_directory)
    # dH = dict(sorted(dH.items(), key=lambda item: item[1]))
    # dW = dict(sorted(dW.items(), key=lambda item: item[1]))
    #print(dH)
    #print(dW)

    training_data, validaion_data = get_data(train_directory, batch_size, validation_split, "both")
    test_data = get_data(test_directory, batch_size, None, None)
    model = create_model(num_classes)
    model.build(input_shape=(None, 256, 256, 3 ))
    model.compile(optimizer = optimizers.Adam(), loss = losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    model.fit(training_data, validation_data= validaion_data, epochs = 20)
    model.save("./model1")
    print("MODEL DONE EVALUATE FOLLOWING")
    print("MODEL DONE EVALUATE FOLLOWING")
    print("MODEL DONE EVALUATE FOLLOWING")
    print("MODEL DONE EVALUATE FOLLOWING")
    model.evaluate(test_data)

    #diagramme über tensorboard



if __name__ == "__main__":
    main()