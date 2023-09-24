import pandas
import numpy as np
import matplotlib.pyplot as plt

df = pandas.read_csv("decay_pt_results.csv")

avgs = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: []
}

acc = df["acc"]
'''
for i, j in enumerate(df["config/gamma"]):
    print(i, j)
    if j < 0.1:
        avgs[0].append(acc[i])
    elif j < 0.2:
        avgs[1].append(acc[i])
    elif j < 0.3:
        avgs[2].append(acc[i])
    elif j < 0.4:
        avgs[3].append(acc[i])
    elif j < 0.5:
        avgs[4].append(acc[i])
    elif j < 0.6:
        avgs[5].append(acc[i])
    elif j < 0.7:
        avgs[6].append(acc[i])
    elif j < 0.8:
        avgs[7].append(acc[i])
    elif j < 0.9:
        avgs[8].append(acc[i])
    elif j < 1:
        avgs[9].append(acc[i])

print(avgs[9])

for l in avgs:
    avgs[l] = sum(avgs[l]) / len(avgs[l])
 '''
x = df["config/decay_pt"]
y = df["acc"]

#find line of best fit
a, b = np.polyfit(x, y, 1)

#add points to plot
#plt.scatter(x, y)

#add line of best fit to plot
#plt.plot(x, a*x+b)

df.plot(x="config/decay_pt", y="acc", marker="o", linewidth=0)
plt.savefig("imgs/decay_pt_results" + ".png")