import matplotlib.pyplot as plt
import numpy as np

minute_list = sorted(list(np.arange(3, 33, step=3)) + list(np.arange(1.5, 31.5, step=3)))

file_test = open("./result/XGBoost_score_test.txt", 'r')
file_validation = open("./result/XGBoost_score_validation.txt", 'r')

lines_test = file_test.readlines()
lines_validation = file_validation.readlines()

scores_test = []
scores_validation = []

for line in lines_test:
    test_score = float(line.split(':')[1])
    scores_test.append(test_score)

for line in lines_validation:
    validation_score = float(line.split(':')[1])
    scores_validation.append(validation_score)

file_test.close()
file_validation.close()

plt.plot(minute_list, scores_test)
plt.plot(minute_list, scores_validation)
plt.legend(["test set", "validation set"])
plt.show()
