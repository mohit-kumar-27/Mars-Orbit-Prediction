import pandas as pd
import numpy as np
from math import cos, sin, tan, sqrt, atan, radians, degrees
import datetime

# This function computes the time differences between consecutive oppositions 
def get_times(data):
    times = [0]
    for i in range(1, 12):
      date1 = datetime.datetime(data[i-1,0],data[i-1,1],data[i-1,2],data[i-1,3],data[i-1,4])
      date2 = datetime.datetime(data[i,0],data[i,1],data[i,2],data[i,3],data[i,4])
      difference = date2 - date1
      num_days = difference.days + difference.seconds/(60*60*24)
      times.append(num_days)
    return np.array(times)

# This function computes longitude angles for each opposition
def get_oppositions(data):
    longitudes = []
    for i in range(12):
        longitude = data[i,5] * 30 + data[i,6] + data[i,7] / 60 + data[i,8] / 3600
        longitudes.append(longitude)
    return np.array(longitudes)

# This function computes the angular error (delta)
def compute_delta(point, longitude):
    theta = 0
    if point[0] > 0:
        if point[1] > 0:
            theta = atan(point[1] / point[0])
        else:
            point[1] *= -1
            theta = radians(360) - atan(point[1] / point[0])
    else:
        point[0] *= -1
        if point[1] > 0:
            theta = radians(180) - atan(point[1] / point[0])
        else:
            point[1] *= -1
            theta = radians(180) + atan(point[1] / point[0])
    return degrees(abs(theta - longitude))

# Q1
def MarsEquantModel(c, r, e1, e2, z, s, times, oppositions):
    errors = np.zeros(12)
    z = radians(z)
    c = radians(c)
    e2 = radians(e2)
    
    for i in range(12):
        beta = s * times[i]
        z = radians((degrees(z) + beta) % 360)

        # y = mx + c equation of the first line intersecting the mars orbit, m = tan(z)
        c1 = e1 * sin(e2) - e1 * cos(e2) * tan(z)

        # To get point of intersection (x,y) we need to solve a quadratic equation Ax^2 + Bx + C = 0, where
        A = 1 + tan(z) ** 2
        B = 2 * c1 * tan(z) - 2 * cos(c) - 2 * tan(z) * sin(c)
        C = c1 ** 2 - 2 * c1 * sin(c) + 1 - r ** 2

        #  (x,y) coordinates of points of intersection(possibly two)
        D = B ** 2 - 4 * A * C
        x1 = (-B + sqrt(D)) / (2 * A)
        x2 = (-B - sqrt(D)) / (2 * A)
  
        y1 = x1 * tan(z) + c1
        y2 = x2 * tan(z) + c1 

        # compute angular error(delta) for both points and take the smaller value
        delta1 = compute_delta([x1, y1], radians(oppositions[i]))
        delta2 = compute_delta([x2, y2], radians(oppositions[i]))

        delta = min(delta1, delta2)
        errors[i] = delta

    max_error = max(errors)
    return errors, max_error


# Q2
def bestOrbitInnerParams(r, s, times, oppositions):
    best_max_opp_err = float('inf')
    best_opp_errs = []
    params = dict()

    for z in np.arange(oppositions[0] - 15, oppositions[0] + 15, 0.1):
      for c in np.arange(135, 155, 0.1):
        for e1 in np.arange(0.8, 1.9, 0.1):
          for e2 in np.arange(c - 10, c + 10, 0.1):
            opp_errs, max_opp_err = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)
            if max_opp_err < best_max_opp_err:
              best_max_opp_err = max_opp_err
              best_opp_errs = opp_errs
              params = {'c':c, 'e1':e1, 'e2':e2, 'z':z}

    return params['c'], params['e1'], params['e2'], params['z'], best_opp_errs, best_max_opp_err


# Q3
def bestS(r, times, oppositions):
    best_max_opp_err = float('inf')
    best_opp_errs = []
    params = dict()

    for s in np.arange(355/687, 365/687, 1/687):
        c, e1, e2, z, opp_errs, max_opp_err = bestOrbitInnerParams(r, s, times, oppositions)
        if max_opp_err < best_max_opp_err:
            best_max_opp_err = max_opp_err
            best_opp_errs = opp_errs
            params = {'s':s}
    
    return params['s'], best_opp_errs, best_max_opp_err


# Q4
def bestR(s, times, oppositions):
    best_max_opp_err = float('inf')
    best_opp_errs = []
    params = dict()

    for r in np.arange(7, 11, 0.1):
        c, e1, e2, z, opp_errs, max_opp_err = bestOrbitInnerParams(r, s, times, oppositions)
        if max_opp_err < best_max_opp_err:
            best_max_opp_err = max_opp_err
            best_opp_errs = opp_errs
            params = {'r':r}
    
    return params['r'], best_opp_errs, best_max_opp_err


# Q5
def bestMarsOrbitParams(times, oppositions):
    best_max_opp_err = float('inf')
    best_opp_errs = []
    params = dict()
    
    for r in np.arange(7, 11, 0.1):
      for s in np.arange(355/687, 365/687, 1/687):
        c, e1, e2, z, opp_errs, max_opp_err = bestOrbitInnerParams(r, s, times, oppositions)
        if max_opp_err < best_max_opp_err:
          best_max_opp_err = max_opp_err
          best_opp_errs = opp_errs
          params = {'r':r, 's':s, 'c':c, 'e1':e1, 'e2':e2, 'z':z}
            
    return params['r'], params['s'], params['c'], params['e1'], params['e2'], params['z'], best_opp_errs, best_max_opp_err


if __name__ == "__main__":

    # Import oppositions data from the CSV file provided
    data = np.genfromtxt(
        "../data/01_data_mars_opposition_updated.csv",
        delimiter=",",
        skip_header=True,
        dtype="int",
    )



    # Extract times from the data in terms of number of days.
    # "times" is a numpy array of length 12. The first time is the reference
    # time and is taken to be "zero". That is times[0] = 0.0
    times = get_times(data)
    assert len(times) == 12, "times array is not of length 12"



    # Extract angles from the data in degrees. "oppositions" is
    # a numpy array of length 12.
    oppositions = get_oppositions(data)
    assert len(oppositions) == 12, "oppositions array is not of length 12"


    # Call the top level function for optimization
    # The angles are all in degrees
    r, s, c, e1, e2, z, errors, maxError = bestMarsOrbitParams(
        times, oppositions
    )

    assert max(abs(errors)) == maxError, "maxError is not computed properly!"
    print(
        "Fit parameters: r = {0:.4f}, s = {1:.4f}, c = {2:.4f}, e1 = {:.4f}, e2 = {:.4f}, z = {:.4f}".format(
            r, s, c, e1, e2, z
        )
    )
    print("The maximum angular error = {0:2.4f}".format(maxError))