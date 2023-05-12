import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
    ### START CODE HERE ### (approx. 4 lines)
        v[f"dW{str(l + 1)}"] = np.zeros(parameters[f"W{str(l + 1)}"].shape)
        v[f"db{str(l + 1)}"] = np.zeros(parameters[f"b{str(l + 1)}"].shape)
        s[f"dW{str(l + 1)}"] = np.zeros(parameters[f"W{str(l + 1)}"].shape)
        s[f"db{str(l + 1)}"] = np.zeros(parameters[f"b{str(l + 1)}"].shape)
    ### END CODE HERE ###

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        ### START CODE HERE ### (approx. 2 lines)
        v[f"dW{str(l + 1)}"] = (
            beta1 * v[f"dW{str(l + 1)}"]
            + (1 - beta1) * grads[f"dW{str(l + 1)}"]
        )
        v[f"db{str(l + 1)}"] = (
            beta1 * v[f"db{str(l + 1)}"]
            + (1 - beta1) * grads[f"db{str(l + 1)}"]
        )
        ### END CODE HERE ###

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        v_corrected[f"dW{str(l + 1)}"] = v[f"dW{str(l + 1)}"] / (1 - beta1**t)
        v_corrected[f"db{str(l + 1)}"] = v[f"db{str(l + 1)}"] / (1 - beta1**t)
        ### END CODE HERE ###

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        ### START CODE HERE ### (approx. 2 lines)
        s[f"dW{str(l + 1)}"] = (
            beta2 * s[f"dW{str(l + 1)}"]
            + (1 - beta2) * grads[f"dW{str(l + 1)}"] ** 2
        )
        s[f"db{str(l + 1)}"] = (
            beta2 * s[f"db{str(l + 1)}"]
            + (1 - beta2) * grads[f"db{str(l + 1)}"] ** 2
        )
        ### END CODE HERE ###

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        s_corrected[f"dW{str(l + 1)}"] = s[f"dW{str(l + 1)}"] / (1 - beta2 ** t)
        s_corrected[f"db{str(l + 1)}"] = s[f"db{str(l + 1)}"] / (1 - beta2 ** t)
        ### END CODE HERE ###

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        ### START CODE HERE ### (approx. 2 lines)
        parameters[f"W{str(l + 1)}"] = parameters[
            f"W{str(l + 1)}"
        ] - learning_rate * v_corrected[f"dW{str(l + 1)}"] / np.sqrt(
            s_corrected[f"dW{str(l + 1)}"] + epsilon
        )
        parameters[f"b{str(l + 1)}"] = parameters[
            f"b{str(l + 1)}"
        ] - learning_rate * v_corrected[f"db{str(l + 1)}"] / np.sqrt(
            s_corrected[f"db{str(l + 1)}"] + epsilon
        )
            ### END CODE HERE ###

    return parameters, v, s