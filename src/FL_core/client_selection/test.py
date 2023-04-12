from scipy.stats import beta
import numpy as np
a=5
b=5
c=[-0.1,-0.3,-0.5,0,0.1,0.2,0.5]
goal=[]
x=np.arange(0.1,1,0.01)
for item in c:
    # goal.append(np.random.beta(var[0], var[1]))
    if(item>=0):
        a+=item
    else:
        b+=abs(item)
    tmp=beta.pdf(x, a, b)
    goal.append(np.argmax(tmp))
    a=2
    b=2