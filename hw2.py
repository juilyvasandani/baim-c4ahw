#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:46:06 2019

@author: juilyvasandani
"""

def q1(N,B):
    '''Write a function to convert an integer N to a string representing that integer in arbitrary base B representation. Assume that numbers are augmented by capital letters and that 1<B<37.'''

    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # ensures that arguments fit code specifications
    assert(N >= 0)
    assert(1<B<37)

    if N<B:
        # when integer is less than base, we can just index the integer from the string 'digits'
        return str(digits[N])
    else:
        # recursive function to calculate each value when integer value is greater than base (uses mod operator and integer division)
        return str(q1(N//B,B)) + digits[(N%B)]

class Customer(object):

    """A customer class with the following properties:

    Attributes:
        id: an integer representing the customer's id.
        gender: a string representing customers's gender.
        days: a list of day of week purchases
        amounts: a list of purchase amounts
    """

    def __init__(self,cid,cgen,cday,camount):
        '''Initialize a customer with id=cid and gender=cgen'''
        self.id=cid
        self.gen=cgen
        self.total=0
        self.days=[cday]
        self.amounts=[camount]

    def addPurchase(self,day,amount):
        '''Add a new purchase'''
        #pass #You need to add your code here
        self.days.append(day)
        self.amounts.append(amount)

    def getTotalSales(self):
        '''Calculate the total sales for the customer'''
        #pass #You need to add your code here
        self.total = sum(self.amounts)
        return self.total

    def __lt__(self,other):
        '''Returns True if self precedes other in terms of the total amount'''
        # didn't need this function so did not need to modify it
        pass #You need to add your code here


    def cust_str(self):
        '''Returns a string represeting a customer object'''
        #pass #You need to add your code here
        return "Customer #{0}\nGender: {1}\nPurchase Days: {2}\nIndividual Sale Amounts: {3}".format(self.id, self.gen, self.days, self.amounts)


class Database(object):

    """A database of customers with the following properties:

    Attributes:
        dir: directory for which the database is created
        customers: a dictionary of customers organized as 'customer id : cutomer object'
    """

    def __init__(self,filedir):
        '''Initialize a database for directory filedir'''
        self.dir=filedir
        self.customers={}
        self.createDatabase(filedir)

    def createDatabase(self,filedir):
        '''Creates a database based on the directory filedir'''
        import shutil, os
        shutil.unpack_archive(filedir+"data1.zip","temp") #unpack data1.zip into 'temp' folder
        filenames = os.listdir("temp") #get all of the file names in the unpacked file

        for fn in filenames: #loop over all the files
            with open("temp/"+fn) as f:# for each file
                self.importFile(f)
            f.close()

        shutil.rmtree("temp")

    def importFile(self,file):
        '''Import one file into the database'''
        file.readline() #first line contains only headers, so we don't need it
        for line in file:
            data=line.split(",") #split the text by ','

            customerID=data[0]
            # JV_NOTES: checks to see if the customer already exists within the database, then appends the new sale/entry into the entity
            if customerID in self.customers:
                self.customers[customerID].addPurchase(data[2],float(data[3]))
            # JV_NOTES: if customer does not belong in the database yet, we create a new entry for them
            else:
                self.customers[customerID]=Customer(data[0],data[1],data[2],float(data[3]))

    def getMaxSaleID(self):
        '''Returns an integers represeting customer id with most sales'''
        #pass #You need to add your code here
        temp_sales  = 0
        for x in self.customers:
            if self.customers[x].getTotalSales() > temp_sales:
                temp_sales = self.customers[x].getTotalSales()

        for x in self.customers:
            if self.customers[x].getTotalSales()==temp_sales:
                return int(x)

    def __str__(self):
        '''Returns a string represeting first four customers in the database'''
        #pass #You need to add your code here
        count = 0
        string = []
        # loop to iterate through the first four customer entries
        for x in self.customers:
            if count < 4:
                # append customer objects into list
                string.append(self.customers[x].cust_str() + '\n\n')
                count += 1
        beans = ''
        # convert list of customer objects into a string
        beans = beans.join(string)
        return beans


def q2(filedir):
    D=Database(filedir)
    return D.getMaxSaleID()

def q3(filedir):
    D=Database(filedir)
    print(D)
