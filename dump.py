import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

y_zero = (y <= 0)

fig, ax = plt.subplots()
ax.plot(x, y, color='blue')
ax.plot(x, y + 0.1, color='red')

ax.fill_between(x, np.min(y), np.max(y), where=y_zero, alpha=0.3, color='green')

plt.show()