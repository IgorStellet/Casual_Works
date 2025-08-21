import numpy as np
def  RungeKutta4(f, xi, ti, tf, n):
    t = np.zeros(n)
    if isinstance(xi, (float, int)):  #True if xi is a list of floats or ints
        x = np.zeros(n)  # x[k] is the solution at time t[k]
    else:
        neq = len(xi)
        x = np.zeros((n, neq))


    x[0] = xi
    #print (x[0])
    #print (x)
    t[0] =ti
    dt = (tf-ti)/float(n)
    dt2 = dt/2.0
    for k in range(n-1):
        #print (x[k,0], x[k,1], t[k])


        K1 = dt*f(x[k], t[k])
        K2 = dt*f(x[k] + 0.5*K1, t[k] + dt2)
        K3 = dt*f(x[k] + 0.5*K2, t[k] + dt2)
        K4 = dt*f(x[k] + K3, t[k] + dt)
        x[k+1] = x[k] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
        t[k+1] = t[k] + dt
    return x, t
