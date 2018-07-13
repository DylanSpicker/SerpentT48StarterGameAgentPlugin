try:
    import tensorflow as tf
    from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose, Concatenate, Dropout, Add, Subtract, MaxPooling2D
    from keras.models import Model
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.optimizers import Adam
    from keras import backend as K
    from keras import metrics
    from keras.utils import to_categorical
except ImportError:
    raise SerpentError("Setup has not been been performed for the ML module. Please run 'serpent setup ml'")

class KerasDQN():
    def __init__(self, input_size, output_size, learning_rate=.0001):
        # Assign Meta Parameters
        self.learning_rate = learning_rate 

        # Build Basic Keras Model
        game_input = Input(shape=input_size)
        C1 = Conv2D(64, 3, padding='same', activation='relu')(game_input)
        MP1 = MaxPooling2D(2)(C1)
        DO1 = Dropout(0.25)(MP1)
        C2 = Conv2D(64, 3, padding='same', activation='relu')(DO1)
        MP2 = MaxPooling2D(2)(C2)
        DO2 = Dropout(0.25)(MP2)

        Flat = Flatten()(DO2)

        D1 = Dense(128, activation='relu')(Flat)
        D2 = Dense(256, activation='relu')(D1)

        action_output = Dense(output_size, activation='linear')(D2)
        self.model = Model(game_input, action_output)
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))