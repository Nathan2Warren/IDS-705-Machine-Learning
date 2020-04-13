
import numpy as np

r = np.load('pred_result_l.npy')
l = r.shape[0]

print (r)
start = 1500
with open('result.txt', 'w') as f:
    for i in range(l):
        f.write(str(i + start)+ ',' + str(r[i]) + '\n')

