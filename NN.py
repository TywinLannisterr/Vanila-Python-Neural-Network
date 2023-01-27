import numpy as np
# size here is (layer_size, previous layer size)
# for example an input of size 20 *20 with first layer size of 12  would result in size bing an array of (12 * 400)
def initialize_parameters(size):
    w = np.random.normal(0, 0.01, size)
    b = np.zeros(size[1]) 
    b = np. array([b])
    return w,b 


def sigmoid(x):
    sig = 1 / ( 1 + np.exp(-x))
    return sig

def relu(x):
    relu =  x.copy()
    for i in range(len(x)) :
        
        if relu[i] > 0:
            relu[i] = relu[i]
        elif relu[i] < 0: 
            relu[i] = 0
    return relu

def linear_forward(previous_layer , weight, offset):
    return np.add(np.matmul(np.transpose(weight),previous_layer), np.transpose(offset)) 

def activation_forward(Input, activation_type):
    
    if activation_type == "relu":
        return relu(Input)
    elif activation_type == "sigmoid":
        return sigmoid(Input)

def model_forward(X, W, B, activation_type):
    
    model_state = []
    p = []
    model_state.append(np.transpose(np.array([X])))
    for i in range(len(W)-1):
        
        p.append(linear_forward(model_state[i], W[i], B[i]))
        model_state.append(activation_forward(linear_forward(model_state[i], W[i], B[i]), activation_type))
    model_state.append(linear_forward(model_state[-1], W[-1], B[-1]))
    return model_state, model_state[-1], p

def softmax(x):
    result = np.ones(x.shape)
    for i in range(len(x)) : 
        result[i] = np.exp(x[i])

    sum = np.sum(np.exp(x))
    return result/sum

def compute_loss(Z , Y ):
    L = []
    for i in range(len(Z)):
         
        sum1 = np.sum(np.exp(Z[i]))
        sum2 = np.sum(Z[i] * np.transpose(np.array([Y[i]])))
        L.append(np.log(sum1) - sum2)
    J = np.sum(L)/ len(Y)
    return J

def linear_backward(layer_num, model_state,w):
    lin_grad_w = np.ones(w.shape)
    #print(w)
    
    z = np.transpose(model_state[layer_num])

    

    lin_grad_w = np.transpose(np.transpose(lin_grad_w) *z) 
    
    
    lin_grad_b = np.ones(model_state[layer_num + 1].shape)
    return lin_grad_w, lin_grad_b

def sigmoid_backward(x):

    return sigmoid(x) * (1 - sigmoid(x))

def relu_backward(x):
    relu_backward = x.copy()
    for i in range(len(x)):
        if relu_backward[i] < 0 : 
            relu_backward[i] = 0
        elif relu_backward[i] >= 0 : 
            relu_backward[i] = 1
    return relu_backward

def activation_backward(model_state,W,layer_num,backprop, activation, pre_model_state):
    
    if activation == "sigmoid":
        [base_grad_w, base_grad_b] = linear_backward(layer_num-1 , model_state,W[layer_num-1])
        grad_w = np.transpose(sigmoid_backward(pre_model_state[layer_num - 1])) * base_grad_w * backprop
        grad_b = np.transpose(sigmoid_backward(pre_model_state[layer_num - 1])) * np.transpose(base_grad_b) * backprop
        return grad_w, grad_b
    elif activation == "relu":
        [base_grad_w, base_grad_b] = linear_backward(layer_num-1 , model_state,W[layer_num-1])
        grad_w = np.transpose(relu_backward(pre_model_state[layer_num - 1])) * base_grad_w * backprop
        grad_b = np.transpose(relu_backward(pre_model_state[layer_num - 1])) * np.transpose(base_grad_b) * backprop
        return grad_w, grad_b

def model_backward( W, model_state, Y_hat, activation, pre_model_state ):
    grad_E = 1/np.sum(np.exp(model_state[-1])) * np.exp(model_state[-1]) 
    grad_E = grad_E - np.transpose(np.array([Y_hat]))
    grad_W = []
    grad_B = []
    a, b = linear_backward(len(model_state) - 2,model_state, W[-1] )
    grad_W.append(a * np.transpose(grad_E) )
    grad_B.append(np.transpose(b * grad_E))
    grad_E = np.matmul(np.transpose(grad_E) , np.transpose(W[-1]))
    for i in reversed(range(len(model_state))):
        
        if i == len(model_state) - 1 :
            continue
        if i == 0 : 
            continue
        [grad_w, grad_b] = activation_backward(model_state,W,i,grad_E,activation, pre_model_state)
        grad_W.append(grad_w) 
        grad_B.append(grad_b) 
        if ( activation == "sigmoid"):
            grad_E = np.matmul(grad_E * np.transpose(sigmoid_backward(model_state[i])) , np.transpose(W[i-1]))
        elif activation == 'relu': 
            grad_E = np.matmul(grad_E * np.transpose(relu_backward(model_state[i])) , np.transpose(W[i-1]))
    grad_W.reverse()
    grad_B.reverse()
    return [grad_W,grad_B]

def update_parameters(W, grad_W , B, grad_B, alpha): 
    for element in range(len(W)):
        
        grad_W[element] = alpha * grad_W[element]
        grad_B[element] = alpha * grad_B[element]
        W[element] = W[element] -  grad_W[element]
        B[element] = B[element] -  grad_B[element]
    
    return W,B 

def predict(Input, trained_W , trained_B , model_activation):
    [model_state, result , pre_model_state]=model_forward(Input, trained_W, trained_B, model_activation)
    softmax_result = softmax(result)
    return softmax_result
     

def random_mini_batches(data,batch_size ):
    batches = []
    flag = 0
    while flag < len(data):
        batch = []
        for i in range(batch_size):
            batch.append(data[i])
            flag += 1 
        batches.append(batch)
    return batches

def train_model(x_train, y_train, model,activation , iterations, learning_rate , batch_size, x_test, y_test):

    W = []
    B = []
    if len(model) == 0: 
        [w0,b0] = initialize_parameters([len(x_train[0]), len(y_train[0])])
        W.append(w0)
        B.append(b0)
    else:
        [w0,b0] = initialize_parameters([len(x_train[0]), model[0]])
        W.append(w0)
        B.append(b0)
        for i in range(len(model)):
            if i == 0:
                continue
            [w,b] = initialize_parameters([model[i-1],model[i]])
            W.append(w)
            B.append(b)
        [W_end, b_end] = initialize_parameters([model[-1], len(y_train[0])])
        W.append(W_end)
        B.append(b_end)
    data = np.concatenate((x_train,y_train) , axis = 1 )
    np.random.shuffle(data)
    y_train_new = []
    for i in data:
        y_train_new.append(i[len(x_train[0]):])
    batches = random_mini_batches(data, batch_size)
    loss_train = []
    loss_test = []
    accuracy_train = []
    accuracy_test = []
    for i in range(iterations):
        Z = []
        Z_test =[]
        for j in batches:
            grad_W = [element * 0 for element in W] 
            grad_B = [element * 0 for element in B]
            for x in j:
                x_i = x[:len(x)-10]
                y_i = x[len(x)-10:]

                model_state, current_output, pre_model_state = model_forward(x_i,W,B,activation)
                Z.append(current_output)
                [grad_W_new, grad_B_new] = model_backward(W,model_state,y_i, activation, pre_model_state)
                for element in range(len(grad_W_new)):
                    grad_W[element] = grad_W[element] + grad_W_new[element]
                    grad_B[element] = grad_B[element] + grad_B_new[element]
            
            for element in range(len(grad_W)):
                grad_W[element] = grad_W[element]/len(j)
                grad_B[element] = grad_B[element]/len(j)
            [W,B] = update_parameters(W,grad_W,B,grad_B,learning_rate)
        print(compute_loss(Z, y_train_new))
        loss_train.append(compute_loss(Z, y_train_new))
        flag = 0
        accuracy = 0
        for i in range(len(x_test)):

            result = predict(x_test[i],W,B,'relu')
            Z_test.append(result)
            if np.argmax(y_test[i]) == np.argmax(result):
                accuracy += 1
            flag += 1
        loss_test.append(compute_loss(Z_test,y_test))
        accuracy_test.append(accuracy/flag)
        accuracy = 0
        flag = 0 
        for i in range(len(Z)):
            if np.argmax(y_train_new[i]) == np.argmax(Z[i]):
                accuracy += 1
            flag += 1 
        accuracy_train.append(accuracy/flag)
    return W, B , accuracy_train, accuracy_test , loss_train, loss_test







