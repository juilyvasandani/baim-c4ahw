#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 10:12:01 2019

@author: juilyvasandani
"""

import collections
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import time

def q1(filedir,outputdir):
    # import network into python
    gM=nx.read_graphml(filedir+"PADGM.graphml")
#    gM=nx.read_graphml("/Users/juilyvasandani/Desktop/PADGM.graphml")

#    nx.draw(gM, with_labels=True, node_size=5, node_color = 'lightblue')

    # build an ordered sequence of the degree nodes for frequency distribution
    seq = ([d for n, d in gM.degree()])
    sequence = sorted(seq)
    degreeCount = collections.Counter(sequence)
    # assign degree value and frequency count to variables by unpacking this into tuples
    deg, cnt = zip(*degreeCount.items())

    # retrieve degree distribution per node
    degrees={x[0]:x[1] for x in gM.degree()}

    fig1 = plt.figure()
    # draw distribution per node
    plt.bar(degrees.keys(), degrees.values(), align = 'center', color = 'navy')
    plt.title("Degree Distribution per Node (Network Inset)")
    plt.ylabel("Node Degree")
    plt.xlabel("Node Number")
    # drawing actual network in inset of graph
    plt.axes([0.4, 0.4, 0.5, 0.5])
    Gcc = sorted(nx.connected_component_subgraphs(gM), key=len, reverse=True)[0]
    pos = nx.spring_layout(gM)
    plt.axis('off')
    nx.draw_networkx_nodes(gM, pos, node_size=20, node_color = 'pink')
    nx.draw_networkx_edges(gM, pos, alpha=0.7)

    # plot a histogram with value vs frequency
    fig2 = plt.figure()
    plt.bar(deg, cnt, width=0.50, color='lightblue')
    plt.title("Frequency Distribution of Node Degree")
    plt.ylabel("Frequency")
    plt.xlabel("Node Degree")
    plt.tight_layout()
    plt.show()
#    plt.savefig(os.path.join(outputdir,'jvasanda_hw5_q1.pdf'))

    import matplotlib.backends.backend_pdf

    with matplotlib.backends.backend_pdf.PdfPages(os.path.join(outputdir,'jvasanda_hw4_q3.pdf')) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)

    return print('Density of network obtained from data is: ',nx.density(gM))

def q2(outputdir):

    # class Item with different functions
    class Item(object):
        def __init__(self, n, v, w):
            self.name = n
            self.value = v
            self.weight = w
        def getName(self):
            return self.name
        def getValue(self):
            return self.value
        def getWeight(self):
            return self.weight
        def __str__(self):
            result = '<' + self.name + ', ' + str(self.value)+ ', ' + str(self.weight) + '>'
            return result

    def value(item):
        return item.getValue()
#    def weightInverse(item):
#        return 1.0/item.getWeight()
#    def density(item):
#        return item.getValue()/item.getWeight()

    def data(n):
        # set seed to generate pseudo-random numbers as data for optimalSolution
        np.random.seed(69)
        names = ["item" + str(i) for i in range(n)]
        values = np.random.random(n)
        weights = np.random.random(n)
        items = []
        for i in range(len(values)):
            items.append(Item(names[i], values[i], weights[i]))
        return items

    def fastMaxVal(toConsider, constraint, memo = {}):
        '''Assumes toConsider a list of items, constraint a weight memo supplied by recursive calls. Returns a tuple of the total value of a solution to the 0/1 knapsack problem and the items orrf that solution.'''
        if (len(toConsider), constraint) in memo:
            result = memo[(len(toConsider), constraint)]
        elif toConsider == [] or constraint == 0:
            result = (0, ())
        elif toConsider[0].getWeight() > constraint:
            # Explore right branch only
            result = fastMaxVal(toConsider[1:], constraint, memo)
        else:
            nextItem = toConsider[0]
            # Explore left branch
            withVal, withToTake =\
                    fastMaxVal(toConsider[1:],
                               constraint - nextItem.getWeight(), memo)
            withVal += nextItem.getValue()
            # Explore right branch
            withoutVal, withoutToTake = fastMaxVal(toConsider[1:], constraint, memo)

            # Choose better branch
            if withVal > withoutVal:
                result = (withVal, withToTake + (nextItem,))
            else:
                result = (withoutVal, withoutToTake)
        memo[(len(toConsider), constraint)] = result

        return result

    def optimalSolution(data,constraint):
        #generate all possible combination of 0,1 and convert them to bool in order to use as an index
        powerset=np.array(list(itertools.product(range(2),repeat=n)),dtype=bool)

        #start with an empty knapsack
        best_combo=[]
        best_value=0
        best_weight=0

        #Go over all possible combinations
        for combo in powerset:

            total_value=sum(list(data['value'][combo]))
            total_weight=sum(list(data['weight'][combo]))

        #if current combination is better than currently known combination, then save it
        if total_weight<=constraint and total_value>best_value:
            best_combo=list(data['item'][combo])
            best_value=total_value
            best_weight=total_weight

        return best_value

    def measuretime(func, args, constraint):
        start = time.time()
        func(args, constraint)
        end = time.time()
        return end - start

    dynamictime = []
    opttime = []
    base = np.random.randint(5,15)

    for n in range(base):
        # set seed to generate pseudo-random numbers
        np.random.seed(69)
        data1=pd.DataFrame({'item':['item'+str(i) for i in range(n)], "value":np.random.random(n), "weight":np.random.random(n)})
        # generate dataset in list form to input into dynamictime function
        data2 = data(n)
        # input similar constraints for both functions
        constraint=np.random.random()*n/2
        opttime.append(measuretime(optimalSolution, data1, constraint))
        dynamictime.append(measuretime(fastMaxVal, data2, constraint))

    plt.plot(range(base),opttime,label='"Brute Force" Optimal Solution')
    plt.plot(range(base),dynamictime,label='Dynammic Programming Solution')
    plt.title('Execution Performance Comparison')
    plt.xlabel('N')
    plt.ylabel('time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir,'jvasanda_hw5_q2.pdf'))
