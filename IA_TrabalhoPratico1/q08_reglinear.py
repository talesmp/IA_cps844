# 08) Agora, utilize N = 10. 
# Posteriormente, procurando os pesos usando Regressao Linear, os use como um vetor de pesos inicial para o Algoritmo de Aprendizagem Perceptron. 
# Execute PLA ate que convirja para o vetor final de pesos que separe completamente todos os pontos dentro da amostra. 
# Entre as opcoes abaixo, qual eh o valor mais proximo do numero medio de iteracoes (mais de 1000 execucoes) que o PLA leva para convergir? 
# (Quando estiver implementando o PLA, escolha um ponto aleatorio para o conjunto classificado incorretamente para cada iteracao).
# a) 1; b) 15; c) 300; d) 5000; e) 10000

############### 1ST PART OF THE SETUP: BASIC FUNCTIONS ###################
import numpy as np

def generate_separable_data(N):
    X = 2*np.random.rand(N, 3)-1
    X[:,0] = 1.0
    xa = 2*np.random.rand() - 1
    ya = 2*np.random.rand() - 1
    xb = 2*np.random.rand() - 1
    yb = 2*np.random.rand() - 1
    xa, xb = min(xa, xb), max(xa, xb)
    xa = xa
    ya = ya
    xb = xb
    yb = yb
    a = (yb-ya)/(xb-xa)
    y = 2*(X[:,2] > ya + (X[:,1] - xa)*a)-1
    b = ya - xa*a
    return X, y, a, b

class LinearRegressionBinaryClassifier:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.w = np.linalg.solve(X.T@X, X.T@y)
        if self.w[2]!=0:
            self.intercept = -self.w[0]/self.w[2]
            self.slope = -self.w[1]/self.w[2]
    
    def predict(self, X):
        return np.sign(X@self.w)

############## 2ND PART OF THE SETUP: IMPLEMENTING THE PLA ################
class PerceptronBinaryClassifier:
    def __init__(self):
        self.w = np.zeros((X.shape[1],1))
    
    def fit(self, X, y, initial_weights=None):
        self.w = np.zeros((X.shape[1],1))
        if not initial_weights is None:
            self.w = initial_weights
        
        self.iter = 0
        while True:
            yhat = self.predict(X)
            misclassifieds = np.nonzero(y != yhat)[0] 
            if len(misclassifieds) == 0:
                return
            i = np.random.choice(misclassifieds)
            self.w = self.w + y[i]*X[i,:]
            self.iter += 1
            if self.iter > 1000:
                print("Couldn't implement the PLA successfully with 1000 iterations at least once out of the 1000 executions of the experiment.")
                return
            
    def predict(self, X):
        return np.sign(X @ self.w)
    

############## SOLVING THE PROBLEM ITSELF ######################
iters = []
#iters_zeros = []
for i in range(1000):
    # Generate data + true target function
    X, y, fa, fb = generate_separable_data(10)

    # Create a classifier based on least squares linear regression
    ls_classifier = LinearRegressionBinaryClassifier()
    ls_classifier.fit(X, y)
    
    # Create a PLA classifier
    pla = PerceptronBinaryClassifier()
    pla.fit(X, y, initial_weights=ls_classifier.w)
    iters.append(pla.iter)

    # pla_zeros = PerceptronBinaryClassifier()
    # pla_zeros.fit(X, y)
    # iters_zeros.append(pla_zeros.iter)
    
print("Mean iterations to converge when using LS estimated weights:", np.mean(iters))
#print("Mean iterations to converge when not using LS estimated weights:", np.mean(iters_zeros))
