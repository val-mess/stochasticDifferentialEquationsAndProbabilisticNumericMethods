import numpy as np
import matplotlib.pyplot as plt

T=1000
N=100
B=np.zeros(N*T)
time=np.arange(0,T,1/N)
a=3
b=5

def brownian(N,T):
    B=np.zeros(N*T)
    for i in range(1,N*T):
        B[i]=B[i-1]+np.sqrt(time[i]-time[i-1])*np.random.normal(0,1)
    return B

for j in range(8):
    plt.figure
    plt.plot(time,brownian(N,T))
    plt.grid()
    #plt.axhline(y=-a,color='r')
    #plt.axhline(y=b,color='r')
    plt.xlabel('Temps $t$')
    plt.title('Simulation de $(B_t)_{t \geqslant 0}$')


def exitTimeBelow(B):
    i=0
    while(B[i]>-a and i<len(B)-1):
        i=i+1
    tau=time[i]
    return(tau)

def exitTimeAbove(B):
    i=0
    while(B[i]<b and i<len(B)-1):
        i=i+1
    tau=time[i]
    return(tau)

def proba(M):
    pBelow=0
    pAbove=0
    esperance=0
    for i in range(M):
        if exitTimeBelow(brownian(N,T))<exitTimeAbove(brownian(N,T)):
            pBelow+=1
            esperance+=exitTimeBelow(brownian(N,T))
        else:
            pAbove+=1
            esperance+=exitTimeAbove(brownian(N,T))
    pAbove=pAbove/M
    pBelow=pBelow/M
    esperance=esperance/M
    return (pBelow,pAbove,esperance)

def probaAbove(M):
    p=0
    for i in range(M):
        if exitTimeBelow(brownian(N,T))>exitTimeAbove(brownian(N,T)):
            p+=1
    p=p/M
    return p

M=1000
#simu=proba(M)
# print('Proba de sortie par -a :',simu[0])
# print('Valeur théorique :',b/(a+b))
# print('Proba de sortie par b :',simu[1])
# print('Valeur théorique :',a/(a+b))
# print('Esperance :',simu[2])
# print('Valeur théorique :',a*b)

