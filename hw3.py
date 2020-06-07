#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:32:58 2019

@author: juilyvasandani
"""

def q1(outputdir):
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # initialize empty list
    summed = []
    ranged = list(range(10,1001,50))
    for z in ranged:
        # generate random int values for x (start), y (stop) - extension?
        x = np.random.randint(100000)
        y = np.random.randint(x+1,100000+1)
        # build random array randnums that produces different values for each iteration
        randnums = np.random.randint(x,y,z)
        # convert randnums into list to append into empty summed list
        randlist = list(randnums)
        # gives us a list of lists of unequal length to plot the sum functions with diff sizes
        summed.append(randlist)

    def sum_norm(s):
        return sum(s)

    def sum_np(s):
        return np.sum(s)

    totalsums = np.zeros(len(summed))
    totalnpsums = np.zeros(len(summed))

    # where sums represents the sum() & np.sum() functions & a represents the list/array to be summed
    def sumtime(sums,a):
        start = time.time()
        sums(a)
        end = time.time()
        return end - start

    # define a separate function for summing numpy that turns the values into an array first??
    def npsumtime(sums,a):
        a = np.asarray(a)
        start = time.time()
        sums(a)
        end = time.time()
        return end - start

    for l,m in enumerate(summed):
        for p,q in enumerate(summed[l]):
            totalsums[l] = sumtime(sum_norm,m)
            totalnpsums[l] = npsumtime(sum_np,m)

    plt.plot(ranged,totalsums,label='sum() function')
    plt.plot(ranged,totalnpsums,label='np.sum() function')
    plt.title('Comparing sum() Functions')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.legend()
    plt.savefig(os.path.join(outputdir,'jvasanda_hw3_q1.pdf'))

def q2(nb,ne,pb,pe):
    from datetime import datetime
    from datetime import timedelta
    import numpy as np

    # do not forget that this is FIFO
    class Queue:
        '''Implementation of the queue abstract data structure using Python lists'''

        def __init__(self):
            self.items = []

        def isEmpty(self):
            # returns a Boolean
            return self.items == []

        def enqueue(self, item):
            # adds an element to the end of the queue
            self.items.insert(0,item)

        def dequeue(self):
            # returns the element in the front of the queue and removes it
            return self.items.pop()

        def size(self):
            return len(self.items)

    class pass_line(object):
        '''This class is relevant because?'''
        def __init__(self, arr, seat):
            self.arr = arr
            self.seat = seat

    q_business = Queue()
    q_economy = Queue()
    # initialize variables to 0
    proc_business = 0
    proc_economy = 0
    wait_business = 0
    wait_economy = 0

    # arrival times for business + economy customers are equally spread over the 2 hours depending on how many customers come in
    arrival_business = list(np.linspace(0,120,nb,dtype=int))
    arrival_economy = list(np.linspace(0,120,nb,dtype=int))

    i = x = 0

    while i <= 120 or proc_economy != 0 or proc_business != 0 or q_business.items != [] or q_economy.items != []:
        if i in arrival_business:
            if arrival_business.count(i) > 1:
                for x in list(range(arrival_business(i) - 1)):
                    q_business.enqueue(pass_line(i,1))
            if q_economy.size == 0 and q_business.size > 0:
                q_economy.enqueue(pass_line(i,1))
            else:
                q_business.enqueue(pass_line(i,1))

        if i in arrival_economy:
            if arrival_economy.count(i) > 1:
                for x in range(arrival_economy(i) - 1):
                    q_economy.enqueue(pass_line(i,0))
            if q_business.size == 0 and q_economy.size > 0:
                q_business.enqueue(pass_line(i,0))
            else:
                q_economy.enqueue(pass_line(i,0))

        if proc_business == 0:
            if q_business.size() > 0:
                obj_b = q_business.dequeue()
                if obj_b.seat == 1:
                    proc_business = pb - 1
                    wait_business += i - obj_b.arr + pb
                else:
                    proc_business = pe - 1
                    wait_economy += i - obj_b.arr + pe

            elif q_economy.size() > 0:
                obj_e = q_economy.dequeue()
                if obj_e.seat == 0:
                    proc_business = pe - 1
                    wait_economy += i - obj_e.arr + pe
                else:
                    proc_business = pb - 1
                    wait_business += i - obj_e.arr + pb

        else:
            proc_business -= 1

        if proc_economy == 0:
            if q_economy.size() > 0:
                obj_e = q_economy.dequeue()
                if obj_e.seat == 0:
                    proc_economy = pe - 1
                    wait_economy += i - obj_e.arr + pe
                else:
                    proc_economy = pb - 1
                    wait_business += i - obj_e.arr + pb
            else:
                if q_business.size() > 0:
                    obj_b = q_business.dequeue()
                    if obj_b.seat == 1:
                        proc_economy = pb - 1
                        wait_business += i - obj_b.arr + pb
                    else:
                        proc_economy = pe - 1
                        wait_economy += i - obj_e.arr + pe
        else:
            proc_economy -= 1

        # adds to the iteration
        i += 1

    time = '%I:%M %p'
    timenow = datetime.strptime("04:00 PM", time) + timedelta(minutes = i)
    averWaitBusiness = int(wait_business/nb)
    averWaitEconomy = int(wait_economy/ne)
    closingTime = str(timenow.strftime(time))
    return averWaitBusiness, averWaitEconomy, closingTime

def q3(filedir, outputdir):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # should we use columns other than total contribution to represent total amount received?
    # what about subtracting refunds?
    df = pd.read_csv(filedir+'P00000001-ALL.csv', index_col = False, usecols = ['contb_receipt_amt','cand_nm'])
    # group by candidate + sum the amounts received
    candidates = df.groupby(['cand_nm']).sum()
    # sort data frame in descending order to highlight top contribution receipt amounts
    candidates = candidates.sort_values(by = ['contb_receipt_amt'], ascending = False)
    # isolate top 8 candidates into separate data frame
    top = candidates[0:9]
    # rest of the candidates to be grouped together under the 'Other' Slice
    sum_other = pd.Series(candidates['contb_receipt_amt'][8:].sum()).rename('Others')
    sum_other.set_axis(['contb_receipt_amt'], axis = 0, inplace=True)
    # append 'Others' to top
    final = top.append(sum_other, ignore_index=False)

    # set candidate names and total amount as labels
    labs = list('\n' + final.index.values + ':\n\n$' + (final['contb_receipt_amt']/10**6).round(1).astype(str) + ' million')
    # select a color palette
    cols = ["#f8c4f3", "#a3f19e", "#d5cef6", "#e1e481", "#eacce7", "#f6d89c", "#9ceced", "#f3ccb7", "#8bf7d3", "#d2e5b0"]
    # choose to highlight the candidates that are women within the top 8 highest contribution amounts
    explode = [0.01,0,0.15,0,0,0.15,0,0.15,0,0]
    plt.figure(figsize=(10,10))

    plt.pie(final['contb_receipt_amt'], counterclock=False, explode=explode, shadow=True, labels=labs, colors=cols, startangle=90)
    plt.title('Figure by jvasanda')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir,'jvasanda_hw3_q3.pdf'))
