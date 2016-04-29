#!/usr/bin/python

# Lekhraj Sharma
# Data Analyst Nanodegree
# P5: Machine Learning Final Project
# Exploring Enron DataSet
# April 2016

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("final_project_dataset.pkl", "r"))

print "Number of persons:", len(enron_data)

print "Number of features:", len(enron_data.values()[1])

print "Number of persons of interest (POI) in dataset:", \
 len([ 1 for x in enron_data.values() if x['poi'] ])

print "Number of non-persons of interest (non-POI) in dataset:", \
 len([ 1 for x in enron_data.values() if not x['poi'] ])

print "Total Money made by Kenneth Lay, Jeffrey Skilling and Andrew Fastow:", \
 enron_data["LAY KENNETH L"]["total_payments"], \
 enron_data["SKILLING JEFFREY K"]["total_payments"], \
 enron_data["FASTOW ANDREW S"]["total_payments"]

print "Number of persons with valid salary in dataset:", \
 len([ 1 for x in enron_data.values() if x['salary'] != "NaN"])

print "Number of persons with valid email address in dataset:", \
 len([ 1 for x in enron_data.values() if x['email_address'] != "NaN"])

print "Number of persons with not valid total_payment in dataset:", \
 len([ 1 for x in enron_data.values() if x['total_payments'] == "NaN"])

print "Percentage of persons with not valid total_payment in dataset:", \
 100.0 * len([ 1 for x in enron_data.values() if x['total_payments'] == "NaN"])/len(enron_data)

print "Number of persons of interest with not valid total_payment in dataset:", \
 len([ 1 for x in enron_data.values() if x['poi'] and x['total_payments'] == "NaN"])

print "Percentage of persons of interest with not valid total_payment in dataset:", \
 100.0 * len([ 1 for x in enron_data.values() if x['poi'] and x['total_payments'] == "NaN"]) \
         /len([ 1 for x in enron_data.values() if x['poi'] ])