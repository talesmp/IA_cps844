# 11) Agora, transforme os N = 1000 dados de treinamento seguindo o vetor de atributos nao-linear:
# (1, x1, x2, x1*x2, x1**2 , x2**2).
# Encontre o vetor w_tilda que corresponde a solucao da regressao linear. 
# Quais das hipoteses a seguir eh a mais proxima que voce encontrou? 
# Neste caso, proximo significa o valor que mais entra em acordo com sua hipotese 
# (existe uma alta probabilidade de estar acordando com um ponto aleatoriamente selecionado).
# Em media algumas execucoes serao necessarias para assegurar uma resposta estavel.

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

X, y = generate_dataset(1000)
Z = transform(X)
lsr = LinearRegressionBinaryClassifier()
lsr.fit(Z, y)
w_tilde = lsr.w

print('\nThe closest answer is (a), once the results are as follows: \ng(x1,x2) = sign( {0:.2f} + ({1:.2f} x1) + ({2:.2f} x2) + ({3:.2f} x1x2) + ({4:.2f} x1^2) + ({5:.2f} x2^2) )\n'.format(w_tilde[0],w_tilde[1],w_tilde[2],w_tilde[3],w_tilde[4],w_tilde[5]))

# ############### IF PLOTTING IS WANTED, UNCOMMENT THIS PART ########
# ################# PLOTTING THE GRAPH #################
# import matplotlib.pyplot as plt

# yhat = lsr.predict(Z)

# below = np.where(y < 0)
# above = np.where(y >= 0)
# plt.scatter(X[below,1], X[below,2])
# plt.scatter(X[above,1], X[above,2])
# plt.legend(["-1", "+1"])
# plt.title("Input data")
# plt.show()

# below = np.where(yhat < 0)
# above = np.where(yhat >= 0)
# plt.scatter(X[below,1], X[below,2])
# plt.scatter(X[above,1], X[above,2])
# plt.legend(["-1", "+1"])
# plt.title("Predicted classes")
# plt.show()

# correct = np.where(y == yhat)
# wrong = np.where(y != yhat)
# plt.scatter(X[correct,1], X[correct,2])
# plt.scatter(X[wrong,1], X[wrong,2])
# plt.legend(["Correctly predicted", "Misprediction"])
# plt.show()