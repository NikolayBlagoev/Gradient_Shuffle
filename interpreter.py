import matplotlib.pyplot as plt
import numpy as np

arrays = []
for i in range(4):
    tmp = []
    with open(f"results/w4p50k1/log_stats_proj_2_{i}.txt","r") as fd:
        print(i)
        for ln in fd.readlines():
            if "LOSS" in ln:
                
                vl = float(ln.split("LOSS:")[1].strip())
                tmp.append(vl)
    print(len(tmp))
    arrays.append(tmp[:14000])

arrays = np.array(arrays)

arrays = np.mean(arrays, axis = 0)
# arrays = np.mean(arrays.reshape(-1, 5), axis=1)
arrays = np.convolve(arrays,[0.1 for _ in range(10)],"valid")
plt.plot(arrays, label="parameter w4p50k1")


# loss3 = []
# with open(f"results/4_full_length_paths/log_loss.txt","r") as fd:
    
#     for ln in fd.readlines():
#         try:
#             loss3.append(float(ln))
#         except ValueError:
#             continue

# loss3 = np.array(loss3)
# loss3 = np.convolve(loss3,[0.1 for _ in range(10)],"valid")
# plt.plot(loss3, label="1 50% path")

arrays3 = []
for i in range(4):
    tmp = []
    with open(f"results/gw4p50k1/log_stats_proj_2_{i}.txt","r") as fd:
        print(i)
        for ln in fd.readlines():
            if "LOSS" in ln:
                
                vl = float(ln.split("LOSS:")[1].strip())
                tmp.append(vl)
    print(len(tmp))
    arrays3.append(tmp[:14000])

arrays3 = np.array(arrays3)

arrays3 = np.mean(arrays3, axis = 0)
# arrays3 = np.mean(arrays3.reshape(-1, 5), axis=1)
arrays3 = np.convolve(arrays3,[0.1 for _ in range(10)],"valid")
plt.plot(arrays3, label="gradient w4p50k1")



arrays4 = []
for i in range(4):
    tmp = []
    with open(f"results/w4p1k4/log_stats_proj_2_{i}.txt","r") as fd:
        print(i)
        for ln in fd.readlines():
            if "LOSS" in ln:
                
                vl = float(ln.split("LOSS:")[1].strip())
                tmp.append(vl)
    print(len(tmp))
    arrays4.append(tmp[:14000])

arrays4 = np.array(arrays4)

arrays4 = np.mean(arrays4, axis = 0)
# arrays4 = np.mean(arrays4.reshape(-1, 5), axis=1)
arrays4 = np.convolve(arrays4,[0.1 for _ in range(10)],"valid")
plt.plot(arrays4,label="baseline")



plt.legend() 
plt.show()