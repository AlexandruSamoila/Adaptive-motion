import numpy as np
import csv


def extractData(datafile, target):
    file = open(datafile)

    csvreader = csv.reader(file)

    header = []
    header = next(csvreader)
    print(header)
    rows = []
    index = 0
    for i, h in enumerate(header):
        if h == target:
            index = i
    for row in csvreader:
        rows.append(row)
    data = []

    for x, row in enumerate(rows):
        dataRow = []
        for j, r in enumerate(row):
            if j in range(index, index+6):
                dataRow.append(float(r))
        data.append(dataRow)

    data = np.array(data)
    file.close()
    return data
