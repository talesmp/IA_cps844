# 06) Utilize N = 100. 
# Use Regressao Linear para encontrar g e avaliar E_in, a fracao de pontos dentro da amostra que foram classificados incorretamente. 
# Repita o experimento 1000 vezes e use o valor medio (guarde as g’s que serao usadas novamente no Problema 7). 
# Qual e o valor medio aproximado de E_in? 
# (aproximado eh a opcao que faz a expressao | sua resposta - dada opcao| proxima a 0. Use esta definicao aqui e sempre).
# a) 0;  b) 0.001; c) 0.01; d) 0.1; e) 0.5


############### 1ST PART OF THE SETUP: BASIC FUNCTIONS ###################
import numpy as np

def generate_separable_data(N):
    X = 2*np.random.rand(N, 2+1)-1
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

#plot_linearly_separable_data(X, y, fb, fa, yhat, ls_classifier.intercept, ls_classifier.slope) 
print("Mean in sample error for N=100 is", np.mean(E_ins))



#OTHER 
# d = 2

# class Dataset:
#     def __init__(self, N):
#         self.X = 2*np.random.rand(N, d+1)-1
#         self.X[:,0] = 1.0
#         X = self.X
#         xa = 2*np.random.rand() - 1
#         ya = 2*np.random.rand() - 1
#         xb = 2*np.random.rand() - 1
#         yb = 2*np.random.rand() - 1
#         xa, xb = min(xa, xb), max(xa, xb)
#         self.xa = xa
#         self.ya = ya
#         self.xb = xb
#         self.yb = yb
#         self.a = (yb-ya)/(xb-xa)
#         self.y = self.evaluate(X)
        
#     def evaluate(self, X):
#         return 2*(X[:,2] > self.ya + (X[:,1] - self.xa)*self.a)-1
        
#     def plot(self, y=None):
#         above = self.X[np.where(self.y < 0)]
#         below = self.X[np.where(self.y >= 0)]
#         plt.scatter(above[:,1], above[:,2], marker="x")
#         plt.scatter(below[:,1], below[:,2], marker="o")
#         if not y is None:
#             wrong = np.where(y != self.y)
#             plt.scatter(self.X[wrong,1], self.X[wrong,2], c="red")
#         plt.plot([self.xa, self.xb], [self.ya, self.yb])
#         plt.show()
        
        
# def fit_ls(X, y):
#     return np.linalg.solve(X.T@X, X.T@y)

# def evaluate(X, w):
#     return np.sign(X@w)

# N = 100
# gs = np.zeros([1000, N])

# total_Ein = 0.0
# for i in range(1000):
#     ds = Dataset(N)
#     w_ls = fit_ls(ds.X, ds.y)
#     total_Ein += np.sum(evaluate(ds.X, w_ls) != ds.y)/N
    
# print(total_Ein/1000)
# #0.039540000000000075; 0.03872000000000005, ...
