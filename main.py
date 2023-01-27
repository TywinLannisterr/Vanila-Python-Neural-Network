import load_mnist
import NN
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib as mlp

[x_train, y_train, x_test, y_test] = load_mnist.load_mnist()
print(len(x_train))
iteration = 50
W,B,acc_train, acc_test , loss_train, loss_test = NN.train_model(x_train,y_train,[], "relu", iteration , 0.1 , 60000, x_test, y_test)
print(W,B)
print(acc_test)
print(acc_train)
print(loss_train)
print(loss_test)
flag = 0
accuracy = 0
for i in range(len(x_test)):

    result = NN.predict(x_test[i],W,B,'relu')
    if np.argmax(y_test[i]) == np.argmax(result):
        accuracy += 1
    flag += 1 
print(accuracy/flag)

plt.plot(np.arange(iteration), loss_train, label= "Train loss")
plt.plot(np.arange(iteration), loss_test, label= "Test loss")

#plt.ylim(0,2000)
#plt.xlim(0,100)
plt.xlabel( 'Iteration')
plt.ylabel('Loss')
plt.title("Loss per iteration")
plt.legend()


plt.show()
np.savetxt('data0.csv', W[0], delimiter=',')
np.savetxt('data1.csv', W[1], delimiter=',')




