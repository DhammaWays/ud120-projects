#!/usr/bin/python

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

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "Number of persons:", len(enron_data) # euivalnet to len(enron_data.keys())

print "Number of features in first person dict:", len(enron_data.values()[1])

print "Number of persons of interest in dataset:", len([ 1 for x in enron_data.values() if x['poi'] ])

poi_names_lines  = tuple(open("../final_project/poi_names.txt", "r"))
print "Total number of names of interest:", len(poi_names_lines)-2 # First two lines are header+blank line

print "Total value of the stock belonging to James Prentice:", enron_data["PRENTICE JAMES"]["total_stock_value"]

print "Number of email messages from Wesley Colwell to persons of interest:", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

print "Value of stock options exercised by Jeffrey Skilling:", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

print "Total Money made by by Kenneth Lay, Jeffrey Skilling and Andrew Fastow:", enron_data["LAY KENNETH L"]["total_payments"], enron_data["SKILLING JEFFREY K"]["total_payments"], enron_data["FASTOW ANDREW S"]["total_payments"]

print "Number of persons with valid salary in dataset:", len([ 1 for x in enron_data.values() if x['salary'] != "NaN"])

print "Number of persons with valid email address in dataset:", len([ 1 for x in enron_data.values() if x['email_address'] != "NaN"])

print "Number of persons with not valid total_payment in dataset:", len([ 1 for x in enron_data.values() if x['total_payments'] == "NaN"])

print "Percentage of persons with not valid total_payment in dataset:", 100.0 * len([ 1 for x in enron_data.values() if x['total_payments'] == "NaN"])/len(enron_data)

print "Number of persons of interest with not valid total_payment in dataset:", len([ 1 for x in enron_data.values() if x['poi'] and x['total_payments'] == "NaN"])

print "Percentage of persons of interest with not valid total_payment in dataset:", 100.0 * len([ 1 for x in enron_data.values() if x['poi'] and x['total_payments'] == "NaN"])/len([ 1 for x in enron_data.values() if x['poi'] ])