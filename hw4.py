#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 21:15:18 2019

@author: juilyvasandani
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import random as rd
import os
import sys
import copy

def bubbleSort(alist):
    # iterates through the length of the list, taking away one each time
    for passnum in range(len(alist)-1,0,-1):
        # iterates through each individual element in the list
        for i in range(passnum):
            if alist[i]>alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp

    return alist


def quickSort(alist):
    # iterates through each index of the list (list of length n needs to stop at n-1 to account for the first index being 0)
    quickSortHelper(alist,0,len(alist)-1)

    return alist

def quickSortHelper(alist,first,last):
    if first<last:

        splitpoint = partition(alist,first,last)

        quickSortHelper(alist,first,splitpoint-1)
        quickSortHelper(alist,splitpoint+1,last)


def partition(alist,first,last):
    # set pivot value to be the first element of list
    pivotvalue = alist[first]

    # leftmark is the index of the number next to our pivot value
    leftmark = first+1
    # rightmark is the index of the last number on our list
    rightmark = last

    done = False
    # while True aka keep running until there is a change
    while not done:
    # test to see if leftmark value is less than pivot and if left index has not reached right index
        while leftmark <= rightmark and alist[leftmark] <= pivotvalue:
            leftmark = leftmark + 1

        # test to see if rightmark value is greater than pivot and if right index has not reached left index
        while alist[rightmark] >= pivotvalue and rightmark >= leftmark:
            rightmark = rightmark -1

        # once rightmark has reached an index < the index of leftmark, change done = True to exit while loop
        if rightmark < leftmark:
            done = True
        else:
            # otherwise, swap the values of both marks into each other's index position
            temp = alist[leftmark]
            alist[leftmark] = alist[rightmark]
            alist[rightmark] = temp

    # swap values to sort properly
    temp = alist[first]
    alist[first] = alist[rightmark]
    alist[rightmark] = temp

    return rightmark

def npsort(alist):
    return np.sort(alist)

def q1(filedir):

    sys.setrecursionlimit(10000)
    x = np.random.randint(1,2000,25)

    # build arrays to store time values
    bubsort = np.zeros(len(x))
    qsort = np.zeros(len(x))
    numpysort = np.zeros(len(x))

    # calcuate the time per function
    def measuretime(func,arg):
        start = time.time()
        func(arg)
        end = time.time()
        return end - start

    for l,m in enumerate(x):
        total = rd.sample(range(1,100000),m)
        total2 = copy.deepcopy(total)
        total3 = copy.deepcopy(total)
        bubsort[l] = measuretime(bubbleSort,total)
        qsort[l] = measuretime(quickSort, total2)
        numpysort[l] = measuretime(npsort,np.array(total3))

    plt.scatter(x,bubsort,label='bubbleSort() function')
    plt.scatter(x,qsort,label='quickSort() function')
    plt.scatter(x,numpysort,label='npsort() function')
    plt.title('Comparing sort() functions')
    plt.xlabel('N')
    plt.ylabel('time')
    plt.legend()
#    plt.show()
    plt.savefig(os.path.join(filedir,'jvasanda_hw4_q1.pdf'))

def q2(guess, epsilon):

    def fg(x):
        return ((np.cos(x))-((x)**0.5))

    def funcderiv(x):
        return (((-1*np.sin(x)) - (0.5*(1/x)**0.5)))

    def bisection(func,eps=.001,nmax=1000):
        '''Find root of continuous function where x1 (low value) and x2 (high value) have opposite signs'''
        low = 0
        high = 4
        n = 0

        if func(low)*func(high) > 0:
            print("No root found. Values have the same sign.")
            return None
        else:
            while (high-low)>eps and n<nmax:
                #print('x1= ',low,'x2= ',high)
                n+=1
                midpoint = (low+high)/2.0
                if func(low)*func(midpoint) > 0:
                    low = midpoint
                else:
                    high = midpoint
            return midpoint

    def newton(func,funcderiv,x0,eps=.001,nmax=1000):
        x1=x0-func(x0)/funcderiv(x0)
        while np.abs(x1-x0)>eps:
            x0=x1
            x1=x0-func(x0)/funcderiv(x0)
            #print("x1=",x1)
        return x1

    bi_soln = bisection(fg,epsilon)
    newton_soln = newton(fg,funcderiv,x0=guess)

    return bi_soln,newton_soln

def greedy(data,category,n,ascending=False):

        #Start with an empty knapsack
#        combo=[]
        total_value=0
        total_weight=0
        constraint=np.random.random()*n/2

        #Sort data by the category/characteristic
        data.sort_values(category,ascending=ascending,inplace=True)
        #print(data) #display sorted data
        #print("*"*50)
        for i in data.index:
            if total_weight+data['weight'].loc[i]<=constraint: #add the item to the knapsack if possible
                #print("add item:",data['item'].loc[i]," -> total_weight=",total_weight+data['weight'].loc[i])
#                total_weight+=data['weight'].loc[i]
                total_value+=data['value'].loc[i]
#                combo.append(data['item'].loc[i])

        return total_value

def optimalSolution(data):
    n=len(data.item)

    #generate all possible combination of 0,1 and convert them to bool in order to use as an index
    powerset=np.array(list(itertools.product(range(2),repeat=n)),dtype=bool)

    #start with an empty knapsack
#    best_combo=[]
    best_value=0
#    best_weight=0

    #Go over all possible combinations
    for combo in powerset:

        total_value=sum(list(data['value'][combo]))
        total_weight=sum(list(data['weight'][combo]))

    #if current combination is better than currently known combination, then save it
    if total_weight<=20 and total_value>best_value:
#        best_combo=list(data['item'][combo])
        best_value=total_value
#        best_weight=total_weight

    return best_value

def q3(filedir):

    # write a function that measures the time of greedt algorithm
    def measuretimegreedy(func,data,category):
        start = time.time()
        func(data,category)
        end = time.time()
        return end - start

    def measuretimeopt(func,args):
        start = time.time()
        func(args)
        end = time.time()
        return end - start

    greedytimevalue = []
    greedytimeweight = []
    greedytimevtw = []
    opttime = []
    base = np.random.randint(5,15)
    accuracy = []
    greedy1 = []
    greedy2 = []
    greedy3 = []
    opt = []
#    counter = 0
    for n in range(base):
        data=pd.DataFrame({'item':['item'+str(i) for i in range(n)], "value":np.random.random(n), "weight":np.random.random(n)})
        data['value-to-weight']=data.value/data.weight
        alist = ['value', 'weight', 'value-to-weight']
        # compare times between greedy algorithm for different categories and optimal solution
        greedytimevalue.append(measuretimegreedy(greedy,data,alist[0],n))
        greedytimeweight.append(measuretimegreedy(greedy,data,alist[1],n))
        greedytimevtw.append(measuretimegreedy(greedy,data,alist[2],n))
        opttime.append(measuretimeopt(optimalSolution,data))
        # test to see accuracy of solutions
        counter = 0
        greedy1.append(greedy(data,alist[0]),n)
        greedy2.append(greedy(data,alist[1]),n)
        greedy3.append(greedy(data,alist[2]),n)
        opt.append(optimalSolution(data))
        if greedy(data,alist[0]) == optimalSolution(data):
            counter += 1
        elif greedy(data,alist[1]) == optimalSolution(data):
            counter += 1
        elif greedy(data,alist[2]) == optimalSolution(data):
            counter += 1
        accuracy.append(counter/base)

    fig1 = plt.figure()
    plt.scatter(range(base),greedytimevalue,label="Greedy 'value' Algorithm")
    plt.scatter(range(base),greedytimeweight,label="Greedy 'weight' Algorithm")
    plt.scatter(range(base),greedytimevtw,label="Greedy 'value-to-weight' Algorithm")
    plt.scatter(range(base),opttime,label='"Brute Force" Optimal Solution')
    plt.title('Time vs. Size of Problem')
    plt.xlabel('N')
    plt.ylim(0,0.020)
    plt.ylabel('time')
    plt.legend()
#    plt.show()

    fig2 = plt.figure()
    plt.plot(range(base),greedy1,label="Greedy 'value' Algorithm")
    plt.plot(range(base),greedy2,label="Greedy 'weight' Algorithm")
    plt.plot(range(base),greedy3,label="Greedy 'value-to-weight' Algorithm")
    plt.plot(range(base),opt,label="Optimal Solution")
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('best value')
    plt.title('Comparing Values Obtained by Greedy Algorithms vs Optimal Solution')
#    plt.show()

    fig3 = plt.figure()
    plt.plot(range(base),accuracy)
    plt.title('Accuracy of Greedy Algorithm vs. Optimal Solution')
    plt.xlabel('N')
    plt.ylabel('accuracy')
#    plt.show()

    import matplotlib.backends.backend_pdf

    with matplotlib.backends.backend_pdf.PdfPages(os.path.join(filedir,'jvasanda_hw4_q3.pdf')) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
