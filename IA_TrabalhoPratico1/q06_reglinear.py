# 06) Utilize N = 100. 
# Use Regressao Linear para encontrar g e avaliar E_in, a fracao de pontos dentro da amostra que foram classificados incorretamente. 
# Repita o experimento 1000 vezes e use o valor medio (guarde as g’s que serao usadas novamente no Problema 7). 
# Qual e o valor medio aproximado de E_in? 
# (aproximado eh a opcao que faz a expressao | sua resposta - dada opcao| proxima a 0. Use esta definicao aqui e sempre).
# a) 0;  b) 0.001; c) 0.01; d) 0.1; e) 0.5


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
        self.intercept = -self.w[0]/self.w[2]
        self.slope = -self.w[1]/self.w[2]
    
    def predict(self, X):
        return np.sign(X@self.w)
   
############## SOLVING THE PROBLEM ITSELF ######################
E_ins = []

for i in range(1000):
    # Generate data + true target function
    X, y, fa, fb = generate_separable_data(100)

    # Create a classifier based on least squares linear regression
    ls_classifier = LinearRegressionBinaryClassifier()
    ls_classifier.fit(X, y)

    # Find in-sample error
    yhat = ls_classifier.predict(X)
    E_in = np.mean(yhat != y)
    E_ins.append(E_in)

print("Mean in sample error for N=100 is", np.mean(E_ins))

# plot_linearly_separable_data(X, y, fb, fa, yhat, ls_classifier.intercept, ls_classifier.slope) 
# ############### 2ND PART OF THE SETUP: PLOTTING ###################
# from matplotlib import pyplot as plt

# def plot_linearly_separable_data(X, y, f_intercept, f_slope, 
#                                  g=None, g_intercept=None, g_slope=None):
#     """
#     X: data
#     y: measured class
#     f: true divider target function
#     g: estimated divider function
#     """
#     plt.title("Linearly separable dataset")
#     # Divide into classes
#     above = np.where(y > 0)
#     below = np.where(y <= 0)
    
#     # Plot positive class
#     plt.scatter(X[above,1], X[above,2], marker="o")
    
#     # Plot misclassified
#     if g is not None:
#         misclassified = np.where(y != g)
#         plt.scatter(X[misclassified,1], X[misclassified,2], marker=".", facecolor=None, s=50, c="black")
    
#     # Plot negative class
#     plt.scatter(X[below,1], X[below,2], marker="x")
    
#     # Plot true target function
#     plt.plot((-1, 1), (f_intercept-f_slope*1, f_intercept+f_slope*1))
    
#     # Plot computed hypothesis function
#     if g_intercept and g_slope:
#         plt.plot((-2, 2), (g_intercept-2*g_slope, g_intercept+2*g_slope))
        
#     plt.legend(["True target divider", "Estimate divider", "Positive class", "Misclassified", "Negative class",])
        
#     plt.xlim((-2, 2))
#     plt.ylim((-2, 2))
#     plt.show()
