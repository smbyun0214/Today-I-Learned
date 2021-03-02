class AverageFilter(object):
    def __init__(self):
        self.prev_avg = 0
        self.k = 0
    
    def get_average(self, x):
        self.k += 1
        
        alpha = (self.k - 1) / self.k
        avg = alpha* self.prev_avg + (1 - alpha)*x

        self.prev_avg = avg
        return avg


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    def get_volt():
        w = 0 + 4*np.random.normal(loc=0, scale=4)
        z = 14.4 + w
        return z

    avg_filter = AverageFilter()

    dt = 0.2
    t = np.arange(0, 10+dt, step=dt)
    num_samples = t.size

    sample_ys = np.zeros((num_samples, 1))
    avg_ys = np.zeros((num_samples, 1))

    for x in range(num_samples):
        sample = get_volt()
        avg = avg_filter.get_average(sample)

        sample_ys[x] = sample
        avg_ys[x] = avg
    
    plt.margins(0, 0)
    plt.plot(t, sample_ys, "r:*", label="Measured")
    plt.plot(t, avg_ys, "o-", fillstyle="none", label="Average")
    plt.legend()
    plt.savefig("fig/average_filter.png", bbox_inches="tight")
