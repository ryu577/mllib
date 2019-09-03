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


