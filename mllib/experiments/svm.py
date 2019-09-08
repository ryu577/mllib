from sklearn.svm import LinearSVC
import numpy as np
from mllib.experiments.toy_classificn_data import Sep1, Sep2
from sympy import symbols, solve, Eq, groebner

def basic_svm(X,y,c=0.001):
    ## We can't plug C=0 here; get value error.
    clf = LinearSVC(random_state=0, tol=1e-5, C=c)
    clf.fit(X,y)
    ## Coefficients and intercept:
    return clf.coef_, clf.intercept_


g, w1, w2, b, a0,a1,a2,beta,k0,k1,k2,u,v = symbols('g w1 w2 b a0 a1 a2 beta k0 k1 k2 u v')

solve([Eq(a0+a1+a2,1),
       Eq(a0+a1+u*a2+2*beta*w1,0),
       Eq(a0+a1+v*a2+2*beta*w2,0),
       Eq(a0-a1+a2,0),
       Eq(w1+w2+b-g-k0**2,0),
       Eq(w1+w2-b-g-k1**2,0),
       Eq(u*w1+v*w2+b-g-k2**2,0),
       Eq(w1**2+w2**2-1,0),
       Eq(a0*(w1+w2+b-g),0),
       Eq(a1*(w1+w2-b-g),0),
       Eq(a2*(u*w1+v*w2+b-g),0)], [g, w1, w2, b, a0,a1,a2,beta,k0,k1,k2,u,v])


gb = groebner([Eq(a0**2+a1**2+a2**2,1),
       Eq(a0**2+a1**2+u*a2**2+2*beta*w1,0),
       Eq(a0**2+a1**2+v*a2**2+2*beta*w2,0),
       Eq(a0**2-a1**2+a2**2,0),
       Eq(w1+w2+b-g-k0**2,0),
       Eq(w1+w2-b-g-k1**2,0),
       Eq(u*w1+v*w2+b-g-k2**2,0),
       Eq(w1**2+w2**2-1,0),
       Eq(a0*(w1+w2+b-g),0),
       Eq(a1*(w1+w2-b-g),0),
       Eq(a2*(u*w1+v*w2+b-g),0),
       Eq(u-v,0)], [u,v, g, a0,a1,a2,beta,k0,k1,k2, b, w1, w2])

w0, w1, b, a0,a1,a2,k0,k1,k2,u,v = symbols('w0 w1 b a0 a1 a2 k0 k1 k2 u v')

## Three points, [[1,1],[-1,-1],[u,v]]; y: [1,-1,1]
soln = solve([
            Eq(a0**2+a1**2+u*a2**2,w0),
            Eq(a0**2+a1**2+v*a2**2,w1),
            Eq(a0**2+a2**2,a1**2),
            Eq(w0+w1+b-1,k0**2),
            Eq(w0+w1-b-1,k1**2),
            Eq(u*w0+v*w1+b-1,k2**2),
            Eq(a0**2*k0**2,0),
            Eq(a1**2*k1**2,0),
            Eq(a2**2*k2**2,0),
            Eq(u,v),
            Eq(u,2)
        ], [u,v, a0,a1,a2,k0,k1,k2, b, w0, w1])

## Two points, [[1,1],[-1,-1]]; y: [1,-1]
gb = groebner([
            Eq(a0**2+a1**2,w0),
            Eq(a0**2+a1**2,w1),
            Eq(a0,a1),
            Eq(w0+w1+b-1,k0**2),
            Eq(w0+w1-b-1,k1**2),
            Eq(a0*k0,0),
            Eq(a1*k1,0)
        ], [u,v, a0,a1,a2,k0,k1,k2, b, w0, w1])


## Three points, simplified form.
gb = groebner([
            Eq(2*a0**2+(1+u)*a2**2,w0),
            Eq((k0/2)**2+1/2,w0),
            Eq( k2**2+2,2*(u+1)*w0 ),
            Eq(a0*k0,0),
            Eq(a2*k2,0)            
        ], [a0,a2,w0, u, k0,k2])

