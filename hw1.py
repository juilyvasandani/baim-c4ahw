#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 10:00:04 2019

@author: juilyvasandani
"""

def q1(path):
    ''' Write a function to find ID of the customer who made the most purchases in terms of total amount spent in data1.zip.'''
    import shutil
    import os
    import pandas as pd
    os.chdir(path)
    # unpack the zip file into a temporary folder in directory
    shutil.unpack_archive(path+'data1.zip','temp')
    data = os.listdir('temp')
    # outer loop to call each element in the directory
    for i in data:
        with open('temp/'+i,'r') as file:
            allfiles = file.read()
            # inner loop to write all lines into a single text file
            for x in allfiles:
                # due to the volume of files, this process MAY take some time, but the code will work
                with open('consolidated.txt','a+') as file:
                    file.write(x)
    # convert txt file into dataframe
    split_file = pd.read_csv('consolidated.txt', sep=",", header = 0)
    # drop irrelevant columns (only need Sales for q1)
    split_file = split_file.drop(['Gender','DayOfWeek'], axis = 1)
    # remove rows that are not numeric
    final = split_file[split_file.SaleAmount != 'SaleAmount']
    # convert Sales column type from str to float
    final['SaleAmount'] = final['SaleAmount'].astype(float)
    # sum all sales by Customer ID to find max amount spent
    ans = final.groupby('CustomerID').sum()
    # sort in descending order to find customer with highest amount spent
    ans = ans.sort_values(by='SaleAmount', ascending = False)
    # Customer ID is now the row name so cast to a list to retrieve value
    list(ans.index)
    customer_id = int(ans.index[0])
    # delete unzipped text files
    for x in os.listdir():
        if x.endswith('.txt'):
            os.remove(x)
    # removes temporary folder
    shutil.rmtree('temp/')
    return print("Customer #",customer_id, "is the customer who made the most purchases, totalling $"+str(round(ans['SaleAmount'][0],2))+".")

def q2(path):
    ''' Write a function to find the maximum one-time purchase amount in data1.zip.'''
    import shutil
    import os
    import pandas as pd
    os.chdir(path)
    # unpack the zip file into a temporary folder in directory
    shutil.unpack_archive(path+'data1.zip','temp')
    data = os.listdir('temp')
    # outer loop to call each element in the directory
    for i in data:
        with open('temp/'+i,'r') as file:
            allfiles = file.read()
            # inner loop to write all lines into a single text file
            for x in allfiles:
                # due to the volume of files, this process MAY take some time, but the code will work
                with open('consolidated.txt','a+') as file:
                    file.write(x)
    # convert consolidated text file into dataframe
    purchase = pd.read_csv('consolidated.txt', sep =",", header = 0)
    # remove extra header rows from the consolidation process
    purchase = purchase[purchase.SaleAmount != 'SaleAmount']
    # convert Sales column type from str to float
    purchase['SaleAmount'] = purchase['SaleAmount'].astype(float)
    # drop irrelevant columns (aka all except Sales)
    purchase = purchase.drop(['Gender','DayOfWeek'], axis = 1)
    # sort in descending order to find maximum purchase amount
    ans = purchase.sort_values(by='SaleAmount', ascending = False)
    # retrieve the maximum one-time purchase value
    purchase_amount = ans.iat[0,1]
    # delete unzipped text files
    for x in os.listdir():
        if x.endswith('.txt'):
            os.remove(x)
    # removes temporary folder
    shutil.rmtree('temp/')
    return print('The maximum one-time purchase amount is $',purchase_amount,' made by the customer with ID#',ans.iat[0,0])

def q3(path):
    '''Write a function to find the fraction of sales to female customers in data1.zip (in terms of number of times sold, not in terms of amount)'''
    import shutil
    import os
    import pandas as pd
    os.chdir(path)
    # unpack the zip file into a temporary folder in directory
    shutil.unpack_archive(path+'data1.zip','temp')
    data = os.listdir('temp')
    # outer loop to call each element in the directory
    for i in data:
        with open('temp/'+i,'r') as file:
            allfiles = file.read()
            # inner loop to write all lines into a single text file
            for x in allfiles:
                # due to the volume of files, this process MAY take some time, but the code will work
                with open('consolidated.txt','a+') as file:
                    file.write(x)
    # convert consolidated text file into dataframe
    purchase = pd.read_csv('consolidated.txt', sep =",", header = 0)
    # remove extra header rows from the consolidation process
    purchase = purchase[purchase.SaleAmount != 'SaleAmount']
    # drop irrelevant columns (aka all except Sales)
    purchase = purchase.drop(['SaleAmount','DayOfWeek'], axis = 1)
    # sum all sales by Customer ID to find max amount spent
    ans = purchase.groupby('Gender').count()
    total = ans.iat[0,0] + ans.iat[1,0]
    final = ans.iat[0,0]/total
    male = ans.iat[1,0]/total
    male = str(round(male*100,2))
    final_ans = str(round(final*100,2))
    # delete unzipped text files
    for x in os.listdir():
        if x.endswith('.txt'):
            os.remove(x)
    # removes temporary folder
    shutil.rmtree('temp/')
    return print('Approximately', round(final,5),'of sales go to female customers, representing', final_ans,'% of all purchases. Men therefore represent approximately',male,'% of all purchases.')

def q4(n):
    '''Write a function that returns a list of integers corresponding to the Nth row of the Pascal’s triangle'''
    # ensure that the input is not a negative number or a float (rounds down)
    n = abs(int(n))
    start = 1
    rows = [1]
    # loop to calculate each subsequent number in triangle and append to list
    for x in range(1,n):
        start = int((start * (n-x))/(x))
        rows.append(start)
    return rows

def q5(path):
    '''Write a function to find the sum of all even numbers written in all files in data2.zip'''
    import shutil
    import os
    os.chdir(path)
    # unpack the zip file into a temporary folder in directory
    shutil.unpack_archive(path+'data2.zip','temp')
    data = os.listdir('temp')
    with open('new.txt','w') as new:
    # loop to consolidate data into one text file
        for i in data:
            with open('temp/'+i, 'r') as file:
               new.write(file.read() + '\n')
    # reading consolidated file through loop to test for evenness + sum together
    with open('new.txt','r') as old:
        summed = 0
        for x in old:
            x = int(x)
            if x%2 == 0:
               summed += x
    # delete unzipped text files
    for y in os.listdir():
        if y.endswith('.txt'):
            os.remove(y)
    # removes temporary folder
    shutil.rmtree('temp/')
    return print('The sum of all even numbers in data2.zip is',summed)
