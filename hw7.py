#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:12:43 2019

@author: juilyvasandani
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def q1(filedir):
    # Reading data files
    zillow = pd.read_csv(filedir+"ZillowData.csv",encoding = "latin-1")
    census = pd.read_csv(filedir+"CensusData.csv")

    # subsetting cols for 1st data set
    zillow = zillow[['RegionID','RegionName','City','State','Metro','CountyName','SizeRank','2010-01']]

    # Merging data set
    df = zillow.merge(census, left_on='RegionName', right_on='Zip Code ZCTA', how='inner')

    # dropping null values
    df = df.dropna()

    # Running regression
    Y=np.array(df.med_hous_price)
    X=np.vstack([np.ones_like(Y),np.array(df.Census_Population_2010)])
    A=np.linalg.inv(np.dot(X,X.T))
    B=np.dot(X,Y.T)
    C=np.dot(A,B)

    return print("I ran an OLS regression method with a step-by-step algorithm. The liltations of OLS is when now of rows are less than the number of cols the transpose of the matrix is not possible.","The coefficient estimates of the regression" , C)


def q2(outputdir):
    #Simulate data and run the regression
    def linearModel(n):
        '''Function to generate a linear model with n coefficients: n-1 indep variables + intercept.'''
        return np.random.rand(n, 1)

    def simulateData(m, beta, eps=1.0/(2**0.5)):
        '''Function to generate 'm' observations for a linear model whose coefficients are given in 'beta'.'''
        n = len(beta) #Get number of coefficients
        X = np.random.rand(m, n) #Get m values for each of the independent variables
        X[:, 0] = 1.0 #The first indep variable is 1
        e = eps*np.random.randn(m, 1) #Get some random noise
        y = np.dot(X,beta) + e #generate the depended variable
        return (X, y)

    def runRegression(X, y):
        '''Function to solve for the coefcient of linear regression'''
        result = np.linalg.lstsq(X, y, rcond=-1)
        b = result[0]
        return b

    #Will write a simple function to measure time.
    #The function takes three arguments: fun, arg1, and arg2.

    import time
    import numpy as np
    def measureTime(fun,arg1,arg2):
        nTimes=50
        times=np.zeros(nTimes)
        for i in range(nTimes):
            startTime = time.time()
            fun(arg1,arg2)
            endTime = time.time()
            times[i]=endTime-startTime
        return np.mean(times)
    # number of observations are varied
    m = 6**np.arange(3,10)
    # number of independent variables are fixed
    N = 8


    times = np.zeros(len(m)) # create an array to store measured times

    for (i, n) in enumerate(m):
        beta_true = linearModel(N)
        (X, y) = simulateData(n, beta_true, eps=0.2)
        t = measureTime(runRegression, X, y)
        times[i] = t


    #Create benchmarks
    linearTime = [times[0]*(n/m[0]) for n in m]
    lognTime = [times[0]*np.log(n/m[0]) for n in m[1:]]
    nlognTime = [times[0]*(n/m[0])*np.log(n/m[0]) for n in m[1:]]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.loglog (m, times, 'k-',linewidth=3,label="Code")

    #Plot benchmarks
    ax1.loglog (m, linearTime, 'b--',label="Linear")
    ax1.loglog (m[1:], lognTime, 'g--',label="Log(n)")
    ax1.loglog (m[1:], nlognTime, 'y--',label="nLog(n)")
    plt.xlabel('Number of Independent Variables')
    plt.ylabel('Time (Seconds)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(outputdir+'jvasanda_hw7_q2.pdf', bbox_inches='tight')

def q3(filedir,outputdir):
    df = pd.read_csv(filedir+"data.csv")
#    df = pd.read_csv("data.csv")
    #Create a random starting point (i.e., generate a uniform random number for each dimension of data)
    data=np.array(df)
    mins=data.min(axis=0)
    maxs=data.max(axis=0)

    K=3

    np.random.seed(69)
    #Step 0: Initial Guess
    mu0= [[np.random.uniform(low=x[0],high=x[1]) for x in zip(mins,maxs)] for i in range(K)]

    df['Cluster']=0

    #get one data point (i.e., row) from the dataframe
    rel_cols=['x', 'y']
    df.loc[1,rel_cols].tolist()

    #Step 1: Create clusters containing points closest in distance to each centroid
    for index,row in df.iterrows():

        p = np.array(row[rel_cols])
        d = np.array([np.linalg.norm(p-mu0[k]) for k in range(K)])
        bestKindex=np.argmin(d)
        df.Cluster.loc[index]=bestKindex

    #Get averages for each column for cluster 0
    [df[col][df.Cluster==0].mean() for col in rel_cols]

    #Step 2: Update the centroids as the means of all points in each cluster.
    mu1= [np.array([df[col][df.Cluster==k].mean() for col in rel_cols]) for k in range(K)]

    diff=sum([np.linalg.norm(mu1[k]-mu0[k]) for k in range(K)])
    n=2
    nmax=100

    while diff>.00001 and n<nmax:
        n+=1

        mu0=mu1

        for index,row in df.iterrows():

            p = np.array(row[rel_cols])
            d = np.array([np.linalg.norm(p-mu0[k]) for k in range(K)])
            bestKindex=np.argmin(d)
            df.Cluster.loc[index]=bestKindex

    mu1= [np.array([df[col][df.Cluster==k].mean() for col in rel_cols]) for k in range(K)]
    diff=sum([np.linalg.norm(mu1[k]-mu0[k]) for k in range(K)])

    from sklearn.cluster import KMeans

    #Determining number of clusters
    nClusters=range(2,11)
    sumDistances=[]
    for n in nClusters:
        kmeans=KMeans(n_clusters=n).fit(df[rel_cols])
        sumDistances.append(kmeans.inertia_) #Proxy for SSE

    fig1 = plt.figure()
    colors=['red','green','cyan']
    clrs=[colors[i] for i in df['Cluster']]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df['x'], df['y'],c=clrs,s=5)
    plt.xlabel("X")
    plt.ylabel("Y")
#    plt.show()

    fig2 = plt.figure()
    plt.plot(nClusters,sumDistances,'-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum Of Distances')
    plt.title('Number of Clusters vs. SSE')
#    plt.show()

    import matplotlib.backends.backend_pdf

    with matplotlib.backends.backend_pdf.PdfPages(os.path.join(outputdir,'jvasanda_hw7_q3.pdf')) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
