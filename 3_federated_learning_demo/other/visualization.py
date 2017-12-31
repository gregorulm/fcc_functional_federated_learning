"""
Visualization script
2017 Fraunhofer-Chalmers Centre for Industrial Mathematics
Gregor Ulm
"""

from numpy import *
import math
import matplotlib.pyplot as plt


def computeAvg(directory, prefixes):

  # open files, read input
  allValues = dict()

  for prefix in prefixes:
    f = open(directory + "/" + prefix + ".csv", "r")

    for line in f:
      #print line
      tmp   = line.split(",")
      epoch = int(tmp[0])
      ms    = int(tmp[1])

      if epoch in allValues.keys():
        allValues[epoch].append(ms)
      else:
        allValues[epoch] = [ms]


  # check that all ms steps are included!
  keys = sorted(allValues.keys())
  for elem in range(keys[0], keys[len(keys)-1], 500):
    assert elem in keys


  # compute mean/min/max per time stamp
  finalVals = dict() # k: epoch, v: avg. time in ms

  for k in allValues.keys():
    if len(allValues[k]) != 5:
      continue
    else:
      vals         = allValues[k]
      minVal       = min(vals)
      maxVal       = max(vals)
      avgVal       = sum(vals) / 5.0
      finalVals[k] = avgVal

  epochs = sorted(list(finalVals.keys()))

  start = finalVals[keys[0]]
  sec   = []

  for epoch in sorted(finalVals.keys()):
    ms = finalVals[epoch]
    sec.append((ms - start) / 1000)

  return (epochs, sec)




# extract data; manually specify directories and file names
(epochs_1, seconds_1) = computeAvg("erl"    , ["1", "2", "3", "4", "5"])
(epochs_2, seconds_2) = computeAvg("erl_nif", ["1", "2", "3", "4", "5"])


# plot results
plt.plot(epochs_1, seconds_1,'b', label="Erlang only")
plt.plot(epochs_2, seconds_2,'r', label="Erlang + NIFs")
plt.legend(loc=2, borderaxespad=1.0)
plt.xlabel("Epoch")
plt.ylabel("Time in s")
plt.title("Concurrent Execution")
plt.show()
