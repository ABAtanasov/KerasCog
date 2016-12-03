# model from Alex's flipflop file

from keras.layers import TimeDistributed, Dense
from keras.models import Sequential

from backend.Networks import leak_recurrent


def model(params)
    model = Sequential()

    # Daniel's way, incorporating leakage:
    model.add(leak_recurrent(input_dim=2, output_dim=params['N_rec'], return_sequences=True, activation='relu'))

    # We want to add rec_noise to our network:
    # Is it better to not use Gaussian?
    # model.add(newGaussianNoise(rec_noise))

    model.add(TimeDistributed(Dense(output_dim=1, activation='linear')))

    # Note I'm not using mse, unlike Daniel's example. Try changing this if training is slow
    model.compile(loss = 'mse', optimizer='Adam', sample_weight_mode="temporal")

    return model
