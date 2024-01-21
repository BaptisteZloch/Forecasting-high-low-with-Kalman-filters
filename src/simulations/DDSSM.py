import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Dynamic State-Space Stochastic Volatility Model
class DDSSM :
    def __init__(self,N,T,p,mu,kappa,theta,xi,rho,S0=100,V0=0.04):
        # paramètre
        self.mu = mu
        self.kappa = kappa
        self.theta=theta
        self.xi = xi
        self.rho = rho
        self.N=N
        self.T=T
        self.p=p
        # Discrétisation temporelle
        dt = T / N
        # Génération de mouvements browniens corrélés
        np.random.seed(2)
        self.Bt = np.random.normal(0, np.sqrt(dt), self.N)
        self.Zt = self.rho * self.Bt + np.sqrt(1 - self.rho**2) * np.random.normal(0, np.sqrt(dt), self.N)
        # Initialisation des vecteurs de prix et de volatilité
        self.S = np.zeros(N)
        self.V = np.zeros(N)
        self.S[0]=S0
        self.S[1]=S0
        self.V[0]=V0
        self.V[1]=V0
        # Simulation du modèle 
        for i in range(2, self.N):
            logV = np.log10(self.V[i-1]) + dt/self.V[i-1]*(self.kappa * (self.theta - self.V[i - 1]) - 1/2*self.xi**2*self.V[i-1]**(p-1) - self.rho*self.xi*self.V[i-1]**(p-0.5) * (self.mu[i-1] - 1/2*self.V[i-1])) + self.rho*self.xi*self.V[i-1]**(p-3/2)*(np.log(self.S[i-1]) - np.log(self.S[i-2])) + self.xi*self.V[i-1]**(p-1)*np.sqrt(dt)*np.sqrt(1-self.rho)*self.Zt[i-1] 
            self.V[i] = 10**logV
            lnS = np.log(self.S[i-1]) + (self.mu[i-1] -self.V[i]/2)*dt + np.sqrt(dt)*np.sqrt(self.V[i])*self.Bt[i]
            self.S[i] = np.exp(lnS)

    def simulate(self):
        return self.V, self.S

    def disp(self):
        df = pd.DataFrame([self.S,self.V,self.Bt,self.Zt],index=['St','Vt','Bt','Zt']).T
        print(df)
        return