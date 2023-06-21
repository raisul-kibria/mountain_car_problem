import numpy as np
import matplotlib.pyplot as plt

tot_bins = 12
position_state_array = np.linspace(-1.5, +1.5, num=tot_bins-1, endpoint=False)
reward = np.zeros_like(position_state_array) + -0.01
reward[-1] = 1
print(position_state_array)
# plt.scatter(position_state_array, reward)
# plt.xlabel("Horizontal Position")
# plt.ylabel("Associated reward")
# plt.show()
