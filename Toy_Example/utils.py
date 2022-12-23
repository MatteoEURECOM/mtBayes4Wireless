import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import multivariate_normal


def texp(t,x):
    if t==1:
        y=np.exp(x)
    else:
        y=(1+(1-t)*x)
        y[y<0]=0
        y=y**(1 / (1 - t))
    return y

def tlog(t,x):
    x=np.asarray(x)
    if t==1.0:
        y=np.log(x)
    else:
        y=(x**(1.0-t)-1.0)/(1.0-t)
    return y


def tgaussian(t,x,mu,sigma):
    p=-((x-mu)**2)/(2*(sigma**2))
    if t<1:
        C=2*np.sqrt(np.pi)*gamma(1/(1-t))/((3-t)*np.sqrt(1-t)*gamma(0.5*(3-t)/(1-t)))
    elif t==1:
        C=np.sqrt(2*np.pi)*sigma
    elif t>1:
        C=np.sqrt(np.pi)*gamma(0.5*(3-t)/(t-1))/(np.sqrt(t-1)*gamma(1/(t-1)))
    return texp(t,p)/C


def plot_funcs():
    for t in np.linspace(1, 2, 4):
        x = np.linspace(-4, 2, 100)
        plt.plot(x, texp(t, x))
    plt.ylim([0, 3])
    plt.show()

    for t in np.linspace(0.1,2,4):
        x=np.linspace(0.1,5,100)
        plt.plot(x,tlog(t,x))
    plt.ylim([-2,3])
    plt.show()


def plot_qgaussian():
    for t in np.linspace(0,2,10):
        x=np.linspace(-10,10,1000)
        plt.plot(x,qgaussian(t,x,1,1))
    plt.show()

def student_t_density(x,mu,Sigma,v=1):
    # Student-t density function as exp_t as in "Expectation Propagation for t-Exponential Family Using Q-Algebra"
    if(not np.isscalar(mu)):
        mu=np.asarray(mu)
        Sigma=np.asarray(Sigma)
        x = np.expand_dims(np.asarray(x),axis=1)
        k=x.shape[0]
        K=np.linalg.inv(v*Sigma)
        Psi=(gamma((v+k)/2.0)/((np.pi*v)**(k/2.0)*gamma(v/2.0)*np.linalg.det(Sigma)**(0.5)))**(-2.0/(v+k))
        g=(Psi*(v+k)/2.0)*(np.dot(mu.T ,np.dot(K, mu))+1)-(v+k)/2.0
        inner_prod=-(np.diag(np.dot(x.T ,np.dot(K, x)))-2*np.diag(np.dot(mu.T ,np.dot(K, x))))*Psi*((v+k)/2.0)
    else:
        k=1
        K=1/(v*Sigma)
        Psi = (gamma((v + k) / 2.0) / ((np.pi * v) ** (k / 2.0) * gamma(v / 2.0) * (np.abs(Sigma) ** (0.5)))) ** ( -2.0 / (v + k))
        g = (Psi * (v + k) / 2.0) * ((mu*K*mu) + 1) - (v + k) / 2.0
        inner_prod=-(x*K*x-2*mu*K*x)*Psi*((v+k)/2.0)
    return(texp(1+2.0/(v+k),inner_prod-g))

def gaussian_density(X,mu,Sigma):
    if (not np.isscalar(mu)):
        m = len(mu)
        sigma2 = np.diag(Sigma)
        X = X - mu.T
        p = 1 / ((2 * np.pi) ** (m / 2) * np.linalg.det(sigma2) ** (0.5)) * np.exp(
            -0.5 * np.sum(X.dot(np.linalg.pinv(sigma2)) * X, axis=1))
    else:
        X = X - mu
        p=np.exp(-0.5*(X**2)/(Sigma**2))/(Sigma*np.sqrt(2*np.pi))
    if (not np.isscalar(p)):
        p[p < 0.0000000001] = 0.0000000001
    elif (p < 0.0000000001):
        p = 0.0000000001
    return p

def student_t_samples(mu,Sigma,v,N):
    '''
    Output:
    Produce M samples of d-dimensional multivariate t distribution
    Input:
    mu = mu (d dimensional numpy array or scalar)
    Sigma = scale matrix (dxd numpy array)
    v = degrees of freedom
    N # of samples to produce
    '''
    if (not np.isscalar(mu)):
        d = len(Sigma)
        g = np.tile(np.random.gamma(v/2.,2./v,N),(d,1)).T
        Z = np.random.multivariate_normal(np.zeros(d),Sigma,N)
    else:
        d=1
        g = np.squeeze(np.tile(np.random.gamma(v / 2., 2. / v, N), (d, 1)).T)
        Z = np.random.normal(0, np.sqrt(Sigma), N)
    return mu + Z/np.sqrt(g)


def test_underflow(n,t):
    p=np.random.random(n)
    #compute t-log plain
    res=tlog(t,np.prod(p))
    log_p=np.sum(np.log(p))
    res_log=tlog(t,np.exp(log_p))
    return res, res_log

'''
TO PLOT STUFF AND CHECK


x=np.linspace(-50,50,20000)
for i in range(1,5):
    plt.plot(x,student_t_density(x,0,i))
    n, bins, patches = plt.hist(student_t_samples(0,i,1,50000),1000,range=[-50,50],density=True)
    plt.show()


x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z=np.asarray([student_t_density([i,j],[0,0],[[1,0.5],[0.5,1]])  for j in y for i in x])
plt.pcolor(X, Y, Z.reshape(100,100))
plt.show()

'''