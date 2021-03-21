class LowPassFilter():
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev_lpf = None

    def get_lpf(self, x):
        if self.prev_lpf is None:
            self.prev_lpf = x
        
        ret = self.alpha*self.prev_lpf + (1 - self.alpha)*x

        self.prev_lpf = ret
        return ret




if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy import io

    low_pass_filter_0_7 = LowPassFilter(0.7)
    low_pass_filter_0_4 = LowPassFilter(0.4)
    low_pass_filter_0_9 = LowPassFilter(0.9)

    num_sampels = 500
    sample_ys = np.zeros((num_sampels, 1))
    lpf_ys_0_7 = np.zeros((num_sampels, 1))
    lpf_ys_0_4 = np.zeros((num_sampels, 1))
    lpf_ys_0_9 = np.zeros((num_sampels, 1))

    sonar_alt = io.loadmat("SonarAlt.mat")["sonarAlt"][0]
    
    for i in range(num_sampels):
        sample = sonar_alt[i]
        lpf_0_7 = low_pass_filter_0_7.get_lpf(sample)
        lpf_0_4 = low_pass_filter_0_4.get_lpf(sample)
        lpf_0_9 = low_pass_filter_0_9.get_lpf(sample)

        sample_ys[i] = sample
        lpf_ys_0_7[i] = lpf_0_7
        lpf_ys_0_4[i] = lpf_0_4
        lpf_ys_0_9[i] = lpf_0_9

    dt = 0.02
    t = np.arange(0, num_sampels*dt, step=dt)

    plt.margins(0, 0)
    plt.plot(t, sample_ys, "r.", markersize=2, label="Measured")
    plt.plot(t, lpf_ys_0_7, "b-", linewidth=1, label="LPF")
    plt.legend()
    plt.savefig("fig/low_pass_filter.png", bbox_inches="tight")

    plt.clf()

    plt.plot(t, lpf_ys_0_4, "r-", linewidth=1, label=r"$\alpha = 0.4$")
    plt.plot(t, lpf_ys_0_9, "r:", linewidth=1, label=r"$\alpha = 0.9$")
    plt.legend()
    plt.savefig("fig/low_pass_filter_alpha.png", bbox_inches="tight")
