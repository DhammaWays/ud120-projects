#!/usr/bin/python

import numpy

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    errors = numpy.fabs(predictions - net_worths)
    combined = numpy.hstack((ages, net_worths, errors))
    combined = combined[combined[:,2].argsort()] # sort on 3rd column errors
    
    cleaned_data = [(combined[i][0], combined[i][1], combined[i][2]) for i in range(int(len(combined)*0.9))]
    
    return cleaned_data

