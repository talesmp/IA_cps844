# 12) Qual o valor mais proximo do erro de classificacao fora da amostra E_out de sua hipotese no Problema 11? 
# (Estime isso gerando um novo conjunto de 1000 pontos e adicione ruıdo, como antes. 
# Em media 1000 execucoes reduzem a varicao em seus resultados).
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

def transform(X):
    Z = np.zeros((X.shape[0], 6))
    Z[:,0] = 1.0
    Z[:,1] = X[:,1]
    Z[:,2] = X[:,2]
    Z[:,3] = X[:,1] * X[:,2]
    Z[:,4] = X[:,1]**2
    Z[:,5] = X[:,2]**2
    return Z


############### SOLVING THE PROBLEM ITSELF ################

E_outs = []

for i in range(1000):                                   #EXECUTING THE EXPERIMENT 1000 TIMES
    X_all, y_all = generate_dataset(2000)               #GENERATING 2000 POINTS, BEING 1000 FOR TRAINING AND 1000 FOR TESTING

    X_train, y_train = X_all[:1000,:], y_all[0:1000]    #TRAINING SET, EQUIVALENT TO LAST PROBLEM
    X_test, y_test = X_all[1000:,:], y_all[1000:]       #TESTING SET, FOR COMPARATION

    Z_train = transform(X_train)
    Z_test = transform(X_test)

    lsr = LinearRegressionBinaryClassifier()
    lsr.fit(Z_train, y_train)

    y_predicted = lsr.predict(Z_test)
    E_out = np.mean(y_test != y_predicted)
    E_outs.append(E_out)
    
print("Mean out of sample error is:", np.mean(E_outs))