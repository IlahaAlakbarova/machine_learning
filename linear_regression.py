import matplotlib.pyplot as plt
from numpy import genfromtxt


initial_b = 0
initial_m = 0


#loading training data
data = genfromtxt('/home/hamid/datasets/data.csv',delimiter=',')

epochs = len(data) * 10
learning_rate =  1 / (epochs * 10)

#f(x) = mx+b
#m = m - learning_rate * partial_derivative
#b = b - learning_rate * partial_derivative

def square_mean_error(data,m,b):
    err = 0
    for i in range(len(data)):
        err += (m*data[i][0] + b - data[i][1])**2

    return err/(2*len(data))

def partial_derivative(m,b,data):
    part_m = 0
    part_b = 0

    for i in range(len(data)):
        part_m += (m*data[i][0] + b - data[i][1])*data[i][0]
        part_b += (m*data[i][0] + b - data[i][1])


    return (part_m/len(data),part_b/len(data))

def training(m,b,learning_rate,data,epochs):
    
    for i in range(epochs):
        par_der = partial_derivative(m,b,data)
        m = m - learning_rate*par_der[0]
        b = b - learning_rate* par_der[1]
        
    return (m,b)


def main():
    print(f"Before running m = {initial_m} b = {initial_b} and error= {square_mean_error(data,initial_m,initial_b)}")

    m,b = training(initial_m,initial_b,learning_rate,data,epochs)

    print(f"After running m = {m} b = {b} and error= {square_mean_error(data,m,b)}")

    plt.scatter(data[:,0],data[:,1])
    plt.plot(data[:, 0], m * data[:, 0] + b,'r')
    plt.show()    

if __name__ == "__main__":
    main()
