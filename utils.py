# utils.py
import numpy as np
import matplotlib.pyplot as plt

def smooth(values, alpha=0.3):
    out = []
    for v in values:
        if not out:
            out.append(v)
        else:
            out.append(alpha*v + (1-alpha)*out[-1])
    return out
# utils.py
import cv2

def risk_color(score):
    if score < 0.3:
        return (0, 200, 0)      # green
    elif score < 0.6:
        return (0, 200, 200)    # yellow
    else:
        return (0, 0, 255)      # red
def plot_scores(scores, name):
    plt.figure(figsize=(8,4))
    plt.plot(scores, label="Escalation Score")
    plt.ylim(0,1)
    plt.xlabel("Time")
    plt.ylabel("Risk")
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    plt.show()