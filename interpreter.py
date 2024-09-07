import matplotlib.pyplot as plt
import numpy as np

arrays = []
for i in range(4):
    tmp = []
    with open(f"results/w4p1k4/log_stats_proj_2_{i}.txt","r") as fd:
        print(i)
        for ln in fd.readlines():
            if "LOSS" in ln:
                
                vl = float(ln.split("LOSS:")[1].strip())
                tmp.append(vl)
    print(len(tmp))
    arrays.append(tmp[:4000])

arrays = np.array(arrays)

arrays = np.mean(arrays, axis = 0)


arrays2 = []
for i in range(4):
    tmp = []
    with open(f"results/w4p75k1/log_stats_proj_2_{i}.txt","r") as fd:
        print(i)
        for ln in fd.readlines():
            if "LOSS" in ln:
                
                vl = float(ln.split("LOSS:")[1].strip())
                tmp.append(vl)
    print(len(tmp))
    arrays2.append(tmp[:4000])

arrays2 = np.array(arrays2)

arrays2 = np.mean(arrays2, axis = 0)


arrays3 = []
for i in range(4):
    tmp = []
    with open(f"results/w4p50k1/log_stats_proj_2_{i}.txt","r") as fd:
        print(i)
        for ln in fd.readlines():
            if "LOSS" in ln:
                
                vl = float(ln.split("LOSS:")[1].strip())
                tmp.append(vl)
    print(len(tmp))
    arrays3.append(tmp[:4000])

arrays3 = np.array(arrays3)

arrays3 = np.mean(arrays3, axis = 0)
arrays = np.convolve(arrays,[0.1 for _ in range(20)],"valid")
arrays2 = np.convolve(arrays2,[0.1 for _ in range(20)],"valid")
arrays3 = np.convolve(arrays3,[0.1 for _ in range(20)],"valid")
plt.plot(arrays)
plt.plot(arrays2)
plt.plot(arrays3)
plt.legend(['w4p100k4','w4p50k1','w4p25k1']) 
plt.show()