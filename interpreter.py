import matplotlib.pyplot as plt
import numpy as np

arrays = []
for i in range(4):
    tmp = []
    with open(f"results/run3/log_stats_proj_2_{i}.txt","r") as fd:
        print(i)
        for ln in fd.readlines():
            if "LOSS" in ln:
                
                vl = float(ln.split("LOSS:")[1].strip())
                tmp.append(vl)
    print(len(tmp))
    arrays.append(tmp[:624])

arrays = np.array(arrays)

arrays = np.mean(arrays, axis = 0)


arrays2 = []
for i in range(4):
    tmp = []
    with open(f"results/run4/log_stats_proj_2_{i}.txt","r") as fd:
        print(i)
        for ln in fd.readlines():
            if "LOSS" in ln:
                
                vl = float(ln.split("LOSS:")[1].strip())
                tmp.append(vl)
    print(len(tmp))
    arrays2.append(tmp[:624])

arrays2 = np.array(arrays2)

arrays2 = np.mean(arrays2, axis = 0)


arrays3 = []
for i in range(4):
    tmp = []
    with open(f"results/run5/log_stats_proj_2_{i}.txt","r") as fd:
        print(i)
        for ln in fd.readlines():
            if "LOSS" in ln:
                
                vl = float(ln.split("LOSS:")[1].strip())
                tmp.append(vl)
    print(len(tmp))
    arrays3.append(tmp[:624])

arrays3 = np.array(arrays3)

arrays3 = np.mean(arrays3, axis = 0)
arrays = np.convolve(arrays,[0.1 for _ in range(10)],"valid")
arrays2 = np.convolve(arrays2,[0.1 for _ in range(10)],"valid")
arrays3 = np.convolve(arrays3,[0.1 for _ in range(10)],"valid")
plt.plot(arrays)
plt.plot(arrays2)
plt.plot(arrays3)
plt.show()