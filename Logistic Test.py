import numpy as np
import matplotlib.pyplot as plt

def logistic(x, alpha=10, midpoint=0.1):
    return np.atan(np.deg2rad(x/3)) / np.atan(np.deg2rad(2000))


# Extended X-axis range
x = np.linspace(-3000, 3000, 700)
y = logistic(x, alpha=8, midpoint=0.45)
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Sigmoid Curve', color='blue')
plt.axvline(0, color='gray', linestyle='--', label='Midpoint (0)')
plt.title('Full Logistic (Sigmoid) Function Curve')
plt.xlabel('Standard Deviation')
plt.ylabel('Penalty')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()