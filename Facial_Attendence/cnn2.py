def CNN(X_train, y_train, X_test, y_test, epoc=10, output=10, filter_count=10):
    import tensorflow as tf
    from tensorflow import keras
    from keras import Sequential
    from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    import numpy as np
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from keras.preprocessing.image import ImageDataGenerator

    # X_train = X_train / 255.0

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
    )

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    model = Sequential()

    # 1st convolutional Layer
    model.add(Conv2D(128, (5, 5), input_shape=X_train.shape[1:], padding="SAME"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # 2st convolutional Layer
    # for i in range (0,filter_count):
    model.add(Conv2D(96, (5, 5), padding="SAME"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5, 5), padding="SAME"))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    # # 3st convolutional Layer
    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(64, activation="relu", use_bias=True))
    model.add(Dropout(0.25))
    model.add(Dense(52, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu", use_bias=True))
    model.add(Dropout(0.15))
    model.add(Dense(28, activation="relu"))
    model.add(Dense(output, activation="softmax"))

    print(model.summary())
    print(len(X_train))

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.001
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # callback = keras.callbacks.EarlyStopping(
    #     monitor="val_loss",
    #     mode="min",
    #     verbose=0,
    #     patience=5,
    # )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # history = model.fit(
    #     X_train,
    #     y_train,
    #     epochs=epoc,
    #     shuffle="True",
    #     validation_split=0.2,
    #     callbacks=[reduce_lr, early_stopping],
    # )
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=32),
        epochs=epoc,
        shuffle="True",
        validation_data=valid_datagen.flow(X_test, y_test),
        # validation_split=0.2,
        callbacks=[reduce_lr, early_stopping],
    )

    X_test = X_test / 255.0
    result = model.evaluate(X_test, y_test, batch_size=10)

    print("test Loss : ", result[0], " , Test Accuracy : ", result[1])

    return model


# (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# # Normalize because rgb values are ranging from 0 to 255

# X_train = tf.keras.utils.normalize(X_train, axis=1)
# X_test = tf.keras.utils.normalize(X_test, axis=1)
# plt.imshow(X_train[0], cmap=plt.cm.binay)

# print("training set : ", X_train.shape, "  Dimention od each image:", X_train[0].shape)

# IMG_SIZE = 28

# X_trainr = np.array(X_train).reshape(-1, IMG_SIZE, 1)
# X_trainr = np.array(X_test).reshape(-1, IMG_SIZE, 1)


# if len(x_train[0].shape) > 1:
#     print("Flattning....")
#     model.add(Flatten(input_shape=x_train[0].shape))

# output_range = int(y_train[y_train.argmax(axis=0)]) + 1
# print("range: ", output_range, " y : ", y_train)
