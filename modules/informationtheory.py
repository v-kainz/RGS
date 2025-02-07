# All about Information Theory

import os
os.environ['JAX_ENABLE_X64'] = '1'
import yaml
import numpy as np
from numpy import exp, sqrt
import scipy.integrate as integrate
from scipy.stats import beta as Beta
from scipy.special import digamma, betaln
from scipy.optimize import minimize

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jacfwd, grad, jacrev

from jax.numpy         import abs     as jabs
from jax.scipy.special import digamma as jdigamma
from jax.scipy.special import betaln  as jbetaln

import warnings
warnings.filterwarnings('ignore')

with open('config/config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)
MaxCount = float(config['constants']['MaxCount'])
tiny = float(config['constants']['tiny'])
COMPRESSION_METHOD = config['switches']['COMPRESSION_METHOD']

if COMPRESSION_METHOD == 'LUT':
    from modules.globals import LUT

class Info:
    """information representation"""
    def __init__(self, mu, la):  # called very often ->  takes long!!!
        total = mu+la
        if total > MaxCount: #limit to a million counts
            mu = MaxCount *mu/total # rescaled
            la = MaxCount *la/total # rescaled
        if mu<-0.999:         #prevent negative counts
            mu = -0.999
        if la<-0.999:
            la = -0.999
        self.mu = mu #Lagrange multiplyer my
        self.la = la #Lagrange multiplyer lambda
        self.pdf = probability_density
        self.H = hamiltonian
        
        self.mean= Beta.mean(mu+1, la+1)
        self.rms = sqrt(Beta.var(mu+1, la+1))
        # information content w.r.t to uniform distribution, measured by KL
        self.information_content = -betaln(mu+1, la+1) + mu*(digamma(mu+1) - digamma(mu+la+2)) + la*(digamma(la+1) - digamma(mu+la+2))
    def __str__(self):
        return "x = "+str(self.mean).ljust(10)[0:10]+" +- "\
                     +str(self.rms).ljust(10)[0:10]+\
        " ("+str(self.mu).rjust(5)+", "+ str(self.la).rjust(5)+")"
    def __add__(self, J):
        return Info(self.mu+J.mu, self.la+J.la)   
    def __sub__(self, J):
        return Info(self.mu-J.mu, self.la-J.la) 
    def __mul__(self, a):
        return Info(a*self.mu, a*self.la)
    def __rmul__(self, a):
        return self*a
    def draw(self):
        return Beta.random_state.beta(self.mu+1,self.la+1) #generate 
    def average(self, f): # average of f(x) over P(x|I)
        return integrate.quad(lambda x: self.pdf(x, self.mu, self.la) * f(x) ,0,1)[0] # integrates over F(x)*pdf(x) between 0 and 1
    def norm(self):
        return self.average(lambda x:1) # integrates over pdf(x)*1 from 0 to 1
    def arr(self):
        return np.array([self.mu, self.la])
    def binned(self, N_bins):
        distr = [integrate.quad(lambda x: self.pdf(x), i/N_bins,(i+1)/N_bins)[0] for i in range(N_bins)] # why is this not normalized?!?!
        return [distr[i]/sum(distr) for i in range(len(distr))]
    def to_list(self):
        return [float(self.mu), float(self.la)] # float for json

def probability_density(x, mu, la):
    return Beta.pdf(x, mu+1, la+1)

def hamiltonian(x, mu, la):
    return -Beta.logpdf(x, mu+1, la+1)

def DeltaInfo(Ic,Io): #what is new in message Ic compared to previous Io
    DeltaI = Ic - Io
    if DeltaI.mu < 0 or DeltaI.la <0: # agent seems to be less sure
        DeltaI = I0                   # regard message as uniformative
    return DeltaI

def KL(IP,IQ): # called very often -> takes long
    """KL for informations """
    if isinstance(IP, Info) and isinstance(IQ, Info):
        muP = IP.mu
        muQ = IQ.mu
        laP = IP.la
        laQ = IQ.la
        res = (muP-muQ)*(digamma(muP+1) - digamma(muP+laP+2)) +\
            (laP-laQ)*(digamma(laP+1) - digamma(muP+laP+2)) +\
            betaln(muQ+1,laQ+1)-betaln(muP+1,laP+1)
        return res
    else:
        raise NotImplementedError('Non-parametrized information class has not been implemented yet.')
        #if isinstance(IP, Info):
        #    IP = info_nonparametrized(IP.binned(), N_bins)
        #if isinstance(IQ, Info):
        #    IQ = info_nonparametrized(IQ.binned(), N_bins)
        # KL for binned distributions:
        #print('IP: ', IP, sum(IP.distr))
        #print('IQ: ', IQ, sum(IQ.distr))
        #print('sum for KL: ', [IP.distr[i]*np.log(IP.distr[i]/IQ.distr[i]) for i in range(N_bins)])
        #res = sum([np.log(IP.distr[i]**IP.distr[i]/(IQ.distr[i]**IP.distr[i])) for i in range(N_bins)]) # log(P^P/Q^P) for numerical stability if P=0 somehwere
        return res

def jKL(muP,laP,muQ,laQ):
    """KL for informations, unpacked and JAXed """
    res = (muP-muQ)*(jdigamma(muP+1) - jdigamma(muP+laP+2)) +\
          (laP-laQ)*(jdigamma(laP+1) - jdigamma(muP+laP+2)) +\
          jbetaln(muQ+1,laQ+1)-jbetaln(muP+1,laP+1)
    return res

def test_KL():
    p = I0#.pdf()
    q = Info(10,5)#.pdf()
    print("Test 1d KLI")
    print("KL(p,p) = ", KL(p,p))
    print("KL(p,q) = ", KL(p,q))
    print("KL(q,p) = ", KL(q,p))
    print("KL(q,q) = ", KL(q,q))
    
def jcut(J,Jcut):# cut away J below Jcut
    return Jcut + ((J-Jcut)+jabs(J-Jcut))/2

@jax.jit
def jLoss(u, v, J):
    """Loss function: KL for moment matching"""
    Jcut = -1+tiny
    mu = jcut(J[0], Jcut)
    la = jcut(J[1], Jcut)
    return mu*u + la*v + jbetaln(mu+1, la+1)    +    (J[0]-mu)**2+(J[1]-la)**2

grad_jLoss = grad(jLoss,          argnums=2)
hess_jLoss = jacfwd(jacrev(jLoss, argnums=2), argnums=2)

@jax.jit
def jgradLoss(u,v, J):
    return grad_jLoss(u,v, J)

@jax.jit
def jhessLoss(u,v, J):
    return hess_jLoss(u,v, J)

def minimize_KL(u, v, mu_start=0, la_start=0):
        try:
            revLoss = lambda J: jLoss(u,v, J)
            jacLoss = lambda J: jgradLoss(u,v, J)
            hesLoss = lambda J: jhessLoss(u,v, J)
            res1 = [mu_start, la_start]
            fun0 = 1
            fun1 = 0
            while fun1 < fun0: 
                res0 = minimize(revLoss, res1, method='trust-exact', jac = jacLoss, hess = hesLoss, tol = 1e-10)
                fun0 = res0.fun # value of revLoss at res0.x
                res0 = res0.x # x for which revLoss is minimal (mu,la)
                res1 = minimize(revLoss, res0, method='trust-ncg', jac = jacLoss, hess = hesLoss, tol = 1e-10)
                fun1 = res1.fun
                res1 = res1.x
            return res1[0], res1[1] # mu, la
        except:
            print('WARNING: update resulted in Nan. Not updating.')
            return mu_start, la_start
      

@jax.jit
def rescale_moment_log(moment, start, end, N):
    """rescale moment from logarithmic scale to linear LUT scale (indices);
    start/end are given logarithmically"""
    index = N/(end-start) * (jnp.log10(moment) - start)
    return index

@jax.jit
def rescale_moment_poly(moment, start, end, N, exp=10):
    """rescale moment from polynomial scale to linear LUT scale (indices);
    start/end are given like they are"""
    a = (start-end)/(start**exp-end**exp)
    d = start - a*start**exp # = v_lim
    # back to np.linspace(v_lim, end, N)
    lin = ((moment-d)/a)**(1/exp)
    # rescale to indices: 0 ... N-1
    index = (lin-d)*(N-1)/(end-d)
    return index

def match(yc, Itruth, Ilie, Istart =Info(0,0), compression_method=COMPRESSION_METHOD):#method='trust-ncg',, method='trust-exact'
    """KL match of yc*Itruth + (1-yc)*Ilie"""
    if yc == 0: # obvious lie <- speaker blushed
        return Ilie
    if yc == 1: # naive update or confession
        return Itruth
    else:
        if compression_method == 'KL_minimization':
            u = yc*(digamma(Itruth.mu+Itruth.la+2)-digamma(Itruth.mu+1))+(1-yc)*(digamma(Ilie.mu+Ilie.la+2)-digamma(Ilie.mu+1))
            v = yc*(digamma(Itruth.mu+Itruth.la+2)-digamma(Itruth.la+1))+(1-yc)*(digamma(Ilie.mu+Ilie.la+2)-digamma(Ilie.la+1))
            mu, la = minimize_KL(u, v, Istart.mu, Istart.la)
            return Info(mu,la) # updated belief 

        elif compression_method == 'moment_matching':
            # match with conservation of first two moments
            mean_h = (Itruth.mu + 1)/(Itruth.mu + Itruth.la + 2)
            var_h = mean_h*(1-mean_h)/(Itruth.mu + Itruth.la + 3)
            mean_noth = (Ilie.mu + 1)/(Ilie.mu + Ilie.la + 2)
            var_noth = mean_noth*(1-mean_noth)/(Ilie.mu + Ilie.la + 3)
            # superposition
            mean = yc*mean_h + (1-yc)*mean_noth
            var = yc*(var_h + mean_h**2) + (1-yc)*(var_noth + mean_noth**2) - mean**2
            mu = mean**2*(1-mean)/var - mean - 1
            la = mean*(1-mean)**2/var + mean - 2
            return Info(mu,la)

        elif compression_method == 'LUT':
            u = yc*(digamma(Itruth.mu+Itruth.la+2)-digamma(Itruth.mu+1))+(1-yc)*(digamma(Ilie.mu+Ilie.la+2)-digamma(Ilie.mu+1))
            v = yc*(digamma(Itruth.mu+Itruth.la+2)-digamma(Itruth.la+1))+(1-yc)*(digamma(Ilie.mu+Ilie.la+2)-digamma(Ilie.la+1))
            u_ind = rescale_moment_log(u, -3, 1.5, LUT.len)
            v_lim = u-np.log(np.exp(u)-1)
            v_ind = rescale_moment_poly(v, 1.000001*v_lim, 32, LUT.len) 
            if (0<u_ind<LUT.len and 0<v_ind<LUT.len): # point is in LUT
                mu_start = 10**(float(LUT.mu([u_ind, v_ind])))-1
                la_start = 10**(float(LUT.la([u_ind, v_ind])))-1
                mu, la = minimize_KL(u, v, mu_start, la_start) # use LUT only as starting point
            else: # points lies outside LUT -> minimize KL explicitely
                mu, la = minimize_KL(u, v, Istart.mu, Istart.la)
            return Info(mu,la)
    
        else:
            raise ValueError(f'compression method {compression_method} not recognized.')

def make_average_opinion(lst, weights, compression_method=COMPRESSION_METHOD):
    """calculate average opinion from list of beta-distributions lst according to weights"""
    if not len(lst)==len(weights):
        raise ValueError(f'list and weights have to be of equal length but have lengths {len(lst)} and {len(weights)}, respectively!')
    if len(lst) == 1:
        return lst[0]
    if not sum(weights) == 1: # renormalize
        weights = [w/sum(weights) for w in weights]
    if all((_.mu, _.la) == (0,0) for _ in lst): # all (0,0) at the beginning:
        return Info(0,0) 
    else: # calculate average opinion
        #via matching of ln(x), ln(1-x)
        lst_u = [digamma(lst[i].mu + lst[i].la + 2) - digamma(lst[i].mu + 1) for i in range(len(lst))]
        lst_v = [digamma(lst[i].mu + lst[i].la + 2) - digamma(lst[i].la + 1) for i in range(len(lst))]

        u = sum([weights[i]*lst_u[i] for i in range(len(lst))])
        v = sum([weights[i]*lst_v[i] for i in range(len(lst))])

        if compression_method == 'LUT':
            u_ind = rescale_moment_log(u, -3, 1.5, LUT.len)
            v_lim = u-np.log(np.exp(u)-1)
            v_ind = rescale_moment_poly(v, 1.000001*v_lim, 32, LUT.len) 
            if 0<=u_ind<=LUT.len and 0<=v_ind<=LUT.len:
                mu = 10**(float(LUT.mu([u_ind, v_ind])))-1
                la = 10**(float(LUT.la([u_ind, v_ind])))-1
            else: 
                mu, la = minimize_KL(u, v, 0, 0)
        else: # point lies outside LUT -> minimize KL explicitely
            mu, la = minimize_KL(u, v, 0, 0)
        
        return Info(mu,la)


def Sc_dist_trues(Sc):
    return exp(-Sc)

def Sc_dist_ratio(Sc):
    return 0.5*Sc*Sc

def Sc_dist_lies(Sc):
    return Sc_dist_ratio(Sc)*Sc_dist_trues(Sc)

# prior information
I0 = Info(0, 0) # prior mu and lambda 
I0mean = I0.mean    # prior mean
I0rms  = I0.rms     # prior rms

# test
# initialize I0 (non-parametrized)
# -> convert to beta
# see what happens: (0,0) or something else?
#I = info_nonparametrized([1]*N_bins, N_bins)
#Ibeta = I.approx_beta()
#print(Ibeta)
#sys.exit()