#model from daniels xor file

from keras.layers import TimeDistributed, Dense
from keras.models import Sequential

from backend.Networks import leak_recurrent

def model(params):

    model = Sequential()
    model.add(leak_recurrent(input_dim=2, output_dim=params['nb_rec'], return_sequences=True, activation='relu' ,noise=0.1))
    # model.add(newGaussianNoise(rec_noise))
    model.add(TimeDistributed(Dense(output_dim=1, activation='linear')))

    model.compile(loss='mse', optimizer='Adam')

    return model