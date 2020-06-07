#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 13:43:28 2019

@author: juilyvasandani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#from bokeh.core.properties import value
#from bokeh.layouts import gridplot
from bokeh.sampledata.us_states import data as states
#from bokeh.models import ColumnDataSource, Legend, Line, HoverTool
#from bokeh.plotting import figure, output_file, output_notebook, show
import holoviews as hv
from holoviews import opts
hv.notebook_extension('bokeh')

def q1(nyears = 10000):
    yearly = []
    for y in range(1,nyears+1):
        xyz_price = 1
        for x in range(1,366):
            prob = np.random.choice([.10, -.05, "bp", 0], p = [0.005, 0.005, 0.0001, 0.9899])
            if prob == '0':
                continue
            elif prob == 'bp':
                break
            else:
                xyz_price += float(prob)
            yearly.append(xyz_price)

    expected_val = sum(yearly)/len(yearly)

    return round(expected_val, 3)

def q1_ext(nyears = 100):
    yearly = []
    for y in range(nyears+1):
        xyz_price = 1
        for x in range(1,366):
            prob = np.random.choice([.10, -.05, "bp", 0], p = [0.005, 0.005, 0.0001, 0.9899])
            if prob == '0':
                 continue
            elif prob == 'bp':
                  break
            else:
                xyz_price += float(prob)
            yearly.append(xyz_price)



    plt.plot(range(len(yearly)),yearly, color = 'navy')
    plt.title("Daily Variation of XYZ Stock Price")
    plt.xlabel("Number of Simulations")
    plt.ylabel("Expected Value (EV)")
#    plt.show()
    plt.savefig(+'jvasanda_hw6_q1_ext.pdf')
    string = 'Find the distribution of EV for XYZ stock across simulations in your working directory.'

    return string

def q2(filedir, outputdir, n_iters = 10000):
    ec = pd.read_csv(filedir+'electoralCollege.csv')
    pw = pd.read_csv(filedir+'probWin.csv')
    #ec = pd.read_csv('electoralCollege.csv')
    #pw = pd.read_csv('probWin.csv')
    tot = pd.merge(ec, pw, on='State')
    tot['ProbWin'] = tot['ProbWin'].str.replace('%',"")

    p1_win = []
    p2_win = []
    for i in tot['ProbWin']:
        p1 = float(int(i)/100)
        p2 = 1-p1
        p1_win.append(p1)
        p2_win.append(p2)

    tot['P1 Win'] = p1_win
    tot['P2 Win'] = p2_win
    tot.drop('ProbWin', axis = 1)

    # initialize a counter to calculate how many times each party wins
    p1s_sims = 0
    p2s_sims = 0
    # initialize an empty list to collect the sum of seats won per party per sim
    p1_votes = []
    p2_votes = []
    for x in range(n_iters):
        p1s = []
        p2s = []
        for y in range(len(tot)):
            win = np.random.choice(['p1','p2'], p = [p1_win[y], p2_win[y]])
            if win == 'p1':
                p1s.append(tot['Votes'][y])
            else:
                p2s.append(tot['Votes'][y])
        if sum(p1s) > 270:
            p1s_sims += 1
        else:
            p2s_sims += 1
        p1_votes.append(sum(p1s))
        p2_votes.append(sum(p2s))

    fig1 = plt.figure()
    plt.bar(range(n_iters), p1_votes, color = 'orange')
    plt.title('Party 1 Seat Distribution')
    plt.xlabel('Simulation Number')
    plt.ylabel('Number of Seats Won')
    plt.show()

    fig2 = plt.figure()
    plt.bar(range(n_iters), p2_votes, color = 'purple')
    plt.title('Party 2 Seat Distribution')
    plt.xlabel('Simulation Number')
    plt.ylabel('Number of Seats Won')
    plt.show()

    import matplotlib.backends.backend_pdf

    with matplotlib.backends.backend_pdf.PdfPages(os.path.join(outputdir,'jvasanda_hw6_q2.pdf')) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)

    return print("The probability of Party 1 winning is ",p1s_sims/n_iters," while the probability of Party 2 winning is ",p2s_sims/n_iters)

def q3(filedir, outputdir, n_iters = 10000):
    ec = pd.read_csv(filedir+'electoralCollege.csv')
    pw = pd.read_csv(filedir+'probWin.csv')

    tot = pd.merge(ec, pw, on='State')
    tot['ProbWin'] = tot['ProbWin'].str.replace('%',"")

    p1_win = []
    p2_win = []
    for i in tot['ProbWin']:
        p1 = float(int(i)/100)
        p2 = 1-p1
        p1_win.append(p1)
        p2_win.append(p2)

    tot['P1 Win'] = p1_win
    tot['P2 Win'] = p2_win
    tot.drop('ProbWin', axis = 1)

    # coerce each column in data frame to a list
    state_name = list(tot['State'])
    state_votes = list(tot['Votes'])
    state_p1_probs = list(tot['P1 Win'])
    state_p2_probs = list(tot['P2 Win'])

    # initialize a counter to calculate how many times each party wins
    p1s_sims = 0
    p2s_sims = 0
    for x in range(n_iters):
        p1s = []
        p2s = []
        for y in range(len(tot)):
            win = np.random.choice(['p1','p2'], p = [p1_win[y], p2_win[y]])
            if win == 'p1':
                p1s.append(tot['Votes'][y])
            else:
                p2s.append(tot['Votes'][y])
        if sum(p1s) > 270:
            p1s_sims += 1
        else:
            p2s_sims += 1

    # set up the map boundaries for each state
    state_xs = [states[code]["lons"] for code in states if code not in ['HI','AK']]
    state_ys = [states[code]["lats"] for code in states if code not in ['HI','AK']]

    # delete outer states in order to keep the map well aligned
    if 'HI' and 'AK' in states.keys():
        del states['HI']
        del states['AK']

    s=tot.iloc[:,0].values
    # set dictionary values for proportions each state votes for winning party
    props = dict()
    for j in s:
        pwin = np.random.choice(['p1','p2'], p = [(p1s_sims/n_iters),(p2s_sims/n_iters)])
        if (pwin == ['p1']):
            props[j] = props.get(j,0) + float(tot.loc[tot['State'] == j]['P1 Win'])
        else:
            props[j] = props.get(j,0) + float(tot.loc[tot['State'] == j]['P2 Win'])

    data = list(states.values())
    final_graph = []
    # set up data in the structure of holoviews
    for x in data:
        x['Votes']=tot.loc[tot['State'] == str(x['name'])]['Votes']
        x['Party 1']=str(int(float(tot.loc[tot['State'] == str(x['name'])]['P1 Win'])*100))+"%"
        x['Party 2']=str(int(float(tot.loc[tot['State'] == str(x['name'])]['P2 Win'])*100))+"%"
        x['Proportions']=props.get(str(x['name']))
        final_graph.append(x)

    choropleth = hv.Polygons(final_graph, ['lons', 'lats'], [('name', 'State'),
                             ('Votes','Votes'),
                             ('Party 1','Probability of Party 1 Winning'),
                             ('Party 2','Probability of Party 2 Winning'),
                             ('Proportions','Proportion Voted for Winning Party')])

    ch=choropleth.options(logz=True, tools=['hover'], xaxis=None, yaxis=None,
                   show_grid=False, show_frame=False, width=575, height=400,
                   colorbar=False, line_color='black',color_index='Proportions',cmap="RdBu")

    hv.save(ch,"jvasanda_hw6_q3.html")
