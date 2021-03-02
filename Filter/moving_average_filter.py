from collections import deque

class MovingAverageFilter():
    def __init__(self, sample_size):
        self.sample_size = sample_size
        self.buffer = deque(maxlen=sample_size)
    
    def get_moving_average(self, x):
        while len(self.buffer) < self.sample_size:
            self.buffer.append(x)

        self.buffer.append(x)
        return sum(self.buffer) / self.sample_size



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy import io

    mov_avg_filter = MovingAverageFilter(sample_size=10)

    num_sampels = 500
    sample_ys = np.zeros((num_sampels, 1))
    mov_avg_ys = np.zeros((num_sampels, 1))

    sonar_alt = io.loadmat("SonarAlt.mat")["sonarAlt"][0]
    
    for i in range(num_sampels):
        sample = sonar_alt[i]
        mov_avg = mov_avg_filter.get_moving_average(sample)

        sample_ys[i] = sample
        mov_avg_ys[i] = mov_avg

    dt = 0.02
    t = np.arange(0, num_sampels*dt, step=dt)

    plt.margins(0, 0)
    plt.plot(t, sample_ys, "r.", markersize=2, label="Measured")
    plt.plot(t, mov_avg_ys, "b", label="Moving Average")
    plt.legend()
    plt.savefig("fig/moving_average_filter.png", bbox_inches="tight")
