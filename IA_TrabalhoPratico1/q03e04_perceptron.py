# 03) Agora, teste N = 100. 
# Em media quantas iteracoes sao necessarias para que o PLA convirja para N = 100 pontos de treinamento? 
# Informe o valor mais proximo ao seu resultado.
# a) 50; b) 100; c) 500; d) 1000; e) 5000

# 04) Qual a opcao mais se aproxima de P[f(x) != g(x)] para N = 100;
# a) 0.001; b) 0.01; c) 0.1; d) 0.5; e)0.8

import random
import matplotlib.pyplot as plt
import numpy as np

#There's an unknown Target Function (target_func) which is is randomly created at __init__
#This Target Function then divides and defines the color of each point in the dataset
#The Perceptron then uses the Candidate Function to chase the Target Function;

#DEFINING THE FUNCTIONS AND LAYING THE BASIS
#BUILDING THE DATASET
def random_point():
    x0, y0 = random.uniform(-1, 1), random.uniform(-1, 1)
    return (x0, y0)

class Dataset:
    def target_func(self, p):                       #1ST STEP ON DEFINING THE COLOR OF EACH POINT
        if self.target_a*p[0] + self.target_b > p[1]:       # A*xp + B > yp
            return -1
        else:
            return 1
    
    def __init__(self, num_points):                 #DEFINING TWO POINTS AT RANDOM AND DEFINING THE LINE
        p0 = random_point()
        p1 = random_point()
        self.target_a = (p1[1] - p0[1]) / (p1[0] - p0[0])   # SLOPE:        A = (y1-y0)/(x1-x0) 
        self.target_b = p0[1] - self.target_a * p0[0]       # INTERCEPT:    B = y0 - A*x0 
        
        self.xs = []
        self.ys = []
        for i in range(num_points):                 #CREATING THE POINTS OF THE DATASET OF SIZE num_points
            xn = random_point()
            self.xs.append(xn)                      #CREATING AND SAVING THE POINT ITSELF
            self.ys.append(self.target_func(xn))    #AND ITS "COLOR", BASED ON -1 OR +1

    def plot(self, ith):                                 #PLOTTING THE TARGET FUNCTION
        cs = ["red" if y > 0 else "blue" for y in self.ys]
        plt.scatter([x[0] for x in self.xs], [x[1] for x in self.xs], c=cs)
        plt.plot((-1, 1), 
                 (-self.target_a+self.target_b, self.target_a+self.target_b))
        plt.gca().set_aspect(1)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.title('Target function at the {0}th execution of the 1000 experiments'.format((ith+1)))
        plt.show()

#DEFINING THE PERCEPTRON 
class PLA:
    def candidate_func(self, p):                    #DEFINING THE CANDIDATE FUNCTION
        return int(np.sign(self.w[0]*1 + self.w[1]*p[0] + self.w[2]*p[1]))  # w0 = b; w1 = a, related to x; w2 is related to y.
        
    def __init__(self, dataset):
        self.w = np.array([0, 0, 0])    #INITIALIZING AN ARRAY/LIST OF WEIGHTS [w0, w1, w2]
        self.dataset = dataset
        
    def fit(self, plot_iters=False):    #RUNNING THE FITTING ROUTINE 
        self.w = np.array([0, 0, 0])    #INITIALIZING WITH [w0, w1, w2] = [0, 0, 0]
        num_iters = 0
        
        while True:
            misclassified_points = []                               #A LIST OF TUPLES FROM THE zip FUNCTION
            for (x, y) in zip(self.dataset.xs, self.dataset.ys):    #GETTING EACH (point, color) PAIR FROM THE DATASET
                if self.candidate_func(x) != y:                     #CHECKING IF APPLYING THE candidate_func TO x RETURNS THE SAME "COLOR"
                    misclassified_points.append((np.array([1, x[0], x[1]]), y)) #SAVING THIS ERROR: [1, x_of_point, y_of_point] AS x AND THE color AS y
            if len(misclassified_points) > 0:                       #IF ERRORS ARE STILL BEING MADE
                num_iters += 1                                      #REGISTER THAT ONE MORE ITERATION WAS NECESSARY
                x, y = random.choice(misclassified_points)          #GETTING, AT RANDOM, A POINT THAT WAS MISCLASSIFIED
                self.w = self.w + y*x                               #AND USE IT TO TWEAK WITH THE WEIGHTS w, CORRECTING IT MAINLY ACCORDING TO y'S SIGNAL; 
                if plot_iters:
                    self.plot()
            else:
                #print("It was necessary", num_iters, "iterations to get it right.") #USED WHEN CHECKING IT INDIVIDUALLY
                return num_iters
        
    def plot(self, ith):
        cs = ["red" if y > 0 else "blue" for y in self.dataset.ys]
        plt.scatter([x[0] for x in self.dataset.xs], [x[1] for x in self.dataset.xs], c=cs)
        y_left = (self.w[1] - self.w[0]) / self.w[2]
        y_right = (-self.w[1] - self.w[0]) / self.w[2]
        plt.plot((-1,1), (y_left, y_right))
        plt.gca().set_aspect(1)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.title('Candidate function found with PLA after {0} iterations at the {1}th execution of the 1000 experiments'.format(num_iters[ith],(ith+1)))
        plt.show()        

#SETTING THE ENVIRONMENT FOR ANSWERING THE QUESTION
n_points_training = 100         #N = NUMBER OF POINTS IN THE TRAINING SET
n_plots_wanted = 0              #NUMBER OF SCATTER PLOTS WANTED THROUGHOUT THE EXPERIMENT
n_exec_experiment = 1000        #GIVEN BY THE REQUIREMENTS OF THE QUESTION
n_points_verification = 1000    #GIVEN BY THE REQUIREMENTS OF THE QUESTION


num_iters = []
n_correct_classified = 0
n_wrong_classified = 0
for i in range(n_exec_experiment): 
    ds = Dataset(num_points=n_points_training)
    pla = PLA(dataset=ds)
    num_iters.append(pla.fit())
    if ((n_plots_wanted!=0) and ((i+1)%(n_exec_experiment//n_plots_wanted) == 0)):
        ds.plot(i)
        pla.plot(i)
    for j in range(n_points_verification): 
        p = random_point()
        if ds.target_func(p) == pla.candidate_func(p):
            n_correct_classified += 1
        else:
            n_wrong_classified += 1
    
mean_iters = np.mean(num_iters)
p_wrong_classif = float(n_wrong_classified)/(n_correct_classified+n_wrong_classified)

print('\nIt takes, on average and approximately, {0:.0f} iterations to achieve no misclassifications (Mean Iters: {1:.4f}).'.format(mean_iters, mean_iters))
print('Therefore, the answer to Question 3 is option (b), given that 100 is the closest option.')

print('\nThe probability that f and g will disagree on their classification is {0:.2f}% (P[f(x)!=g(x)] = {1:.4f})'.format((p_wrong_classif*100),p_wrong_classif))
print('Therefore, the answer to Question 4 is option (b), given that 0.01 is the closest option.\n')