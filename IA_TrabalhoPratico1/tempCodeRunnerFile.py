############### IF PLOTTING IS WANTED, UNCOMMENT THIS PART ########
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