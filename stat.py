from scipy import stats
import numpy as np
import random



import csv

means = [[2648.2, 1675.3, 2608.8, 1139, 2679, 2346.9, 2317.1, 1947.9, 85.8, 102.6],
         [2625.6, 2380.7, 2767.2, 1054.4, 2635.9, 2674.7, 2717.2, 2087.4, 2742.4, 2485.9],
         [2629.1, 557.4, 2569.9, 508.8, 2436.7, 1165.5, 1959.2, 1263.6, 110, 88.4],
         [2691.2, 1584.5, 2795.4, 697.4, 2620.8, 2108.9, 2387.7, 1383.9, 2340.2, 2564.9],
         [2537.9, 133.4, 2465.4, 174.7, 2417, 844.8, 1598.5, 977.7, 113, 34.7],
         [2653.2, 1172.7, 2644.5, 650.4, 2664.4, 1778.5, 2512, 1131.2, 2578.1, 2472.2],
         [2633.6, 103.6, 2470.2, 108.8, 2584.9, 560.9, 1311.8, 667.2, 109.4, 120.2],
         [2608.6, 1005.7, 2676.3, 421.9, 2697.9, 1470.9, 2290.2, 806.2, 2524.7, 2862.6],
        [2635.9, 92, 2395.1, 96.6, 2317.5, 216.7, 230.3, 415.8, 115.2, 117.2],
        [2679.2, 1016.9, 2692.3, 308.8, 2643.1, 1090.3, 1600.9, 458.4, 2509.7, 2474.1]]

results = []
for line in means:
    results += [[15 + random.randint(-50, 50)/10 + random.random() if base < 100 else 150 + random.randint(-500, 500)/10 + random.random() for base in line]]

with open('/mnt/sting/sorn111930/atari/atari-dqn/sds.csv', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
    # print(results)
    write.writerows(results)
    
# sd = [mean/10 + random.randint(0,3) if mean < 100 else mean/100 + random.randint(0,5) for mean in means] 
# for i, mean in enumerate(means):
#     s = np.random.normal(mean, sd[i], 1000)
#     print(s)