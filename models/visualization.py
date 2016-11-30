def get_activations(model, layer, X_batch):
    get_activations = theano.function([model.layers[0].input], model.layers[layer].output, allow_input_downcast=True)
    activations = get_activations(X_batch)
    return activations


def get_weights(model):
    return model.layers[0].get_weights()


def get_maps(model, X_batch, layer = 0):
    activations = get_activations(model, layer, X_batch)
    return activations == 0


def plot_eig(model, maps, t, condition = 0, color = [0, 0, 1]):
    import scipy.linalg as la
    
    w = get_weights(model)
    wrec = w[1]
    bin_maps = maps.astype('int')
    bin_t = bin_maps[condition, t, :].reshape(np.shape(bin_maps)[2], 1)
    bin_mask = bin_t.dot(bin_t.T)
    eva = la.eig(wrec * bin_mask)
    plt.scatter(eva[0].real, eva[0].imag, 12, color)
    plt.xlabel('real')
    plt.ylabel('imaginary')


# Dave's visualization function
def visualize_rnn_1(filename):

    X,y = create_input_output_pair(0, 1, 2000, 5000);
    
    plt.plot(range(5000), X[:,0], 'y', range(5000), X[:,1], 'r',range(5000), y[:,0], 'b',range(5000), y[:,1], 'g')
    plt.show()

    plt.plot(range(5000), y[:,0], 'b',range(5000), y[:,1], 'g')
    plt.show()

    print 1
    X = np.expand_dims(X, 0)
    print 2

    
    model = rnn_1(filename)
    print 3
    out = model.predict(X)
    print 4

    plt.plot(range(5000), out[0,:,0], 'b',range(5000), out[0,:,1], 'g')
    plt.show()

    return
