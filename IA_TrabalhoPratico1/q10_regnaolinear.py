# 10) Execute a Regressao Linear sem transformacao usando o vetor de atributos: 
# (1, x1, x2), para encontrar o peso w. 
# Qual e o valor aproximado de classificacao do erro E_in dentro da amostra? 
# (Execute o experimento 100 vezes e use o valor medio de E_in para reduzir a variacao nos seus resultados.)
# a) 0; b) 0.1; c) 0.3; d) 0.5; e) 0.8

import numpy as np

class LinearRegressionBinaryClassifier:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.w = np.linalg.solve(X.T@X, X.T@y)
        self.intercept = -self.w[0]/self.w[2]
        self.slope = -self.w[1]/self.w[2]
    
    def predict(self, X):
        return np.sign(X@self.w)

def f(x):
    return np.sign(x[:,1]**2 + x[:,2]**2 - 0.6)

def generate_dataset(N):
    X = np.random.uniform(-1, 1, (N,3))
    X[:,0] = 1.0
    y = f(X)
    
    # Flip 10% to simulate noise
    i_flip = np.arange(0, N)
    np.random.shuffle(i_flip)
    i_flip = i_flip[0:(N//10)]
    y[i_flip] *= -1
    
    return X, y

E_ins = []
for i in range(100):                            #EXECUTING THE EXPERIMENT 100 TIMES
    X, y = generate_dataset(1000)               #EACH EXECUTION HAS 1000 POINTS
    lsr = LinearRegressionBinaryClassifier()
    lsr.fit(X, y)
    yhat = lsr.predict(X)
    E_ins.append(np.mean(y != yhat))
    
print("\nThe mean error rate is {:.4f} ".format(np.mean(E_ins)))
print('Therefore, the answer to Question 10 is option (d), given that 0.5 is the closest option.\n')

# ####### PLOTTING THE GRAPH ########
# import matplotlib.pyplot as plt
# i_above = np.where(y >= 0)
# i_below = np.where(y < 0)
# plt.title("Circular dataset")
# plt.scatter(X[i_below,1], X[i_below,2])
# plt.scatter(X[i_above,1], X[i_above,2])
# plt.legend(["-1", "+1"])
# plt.show()