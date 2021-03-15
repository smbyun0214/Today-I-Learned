## ver 1.0 
import matplotlib.pyplot as plt
import time, math
import scipy.signal as sig
import numpy as np

class filter():
    time_list = np.array([])
    orig_data = None
    prev_data = None
    tau = None
    Ts = None
    
    def __init__(self, f_cut, freq, num_data=1, title='noname'): 
        temp = []
        for i in range(num_data):
            temp.append(0)
        self.prev_data = np.array([temp])
        self.orig_data = np.array([temp])

        self.Ts = 1.0/freq
        w_cut = 2*np.pi*f_cut
        self.tau = 1/w_cut
        self.title = title
        self.Fs = 20*10**3               # 20kHz
        self.f_cut = f_cut
        self.num_data = num_data

    def toHz(self, value):
        from numpy import pi
        return value/2/pi
    
    def LPF(self, sig):
        self.time_list = np.append(self.time_list, np.array([time.time()]))
        prev_sig = (self.prev_data[-1,:])
        value = (self.tau*prev_sig + self.Ts*sig)/(self.tau + self.Ts)
        self.orig_data = np.append(self.orig_data, np.array([sig]), axis=0)
        self.prev_data = np.append(self.prev_data, np.array([value]), axis=0)
        return value

    def plot(self):
        fig = plt.figure()
        num = np.shape(self.prev_data)[1]
        for i in range(num):
            ax = fig.add_subplot(1,num, i+1)
            ax.plot(self.time_list, self.orig_data[1:,i])
            ax.plot(self.time_list, self.prev_data[1:,i])
        plt.title('Filtered Signal : ' + self.title + ', cutoff : ' + str(self.f_cut))
        plt.savefig(self.title+'_output')        

    def draw_FFT_Graph(self, inputData, data, fs, **kwargs):
        from numpy.fft import fft
        import matplotlib.pyplot as plt
        
        graphStyle = kwargs.get('style', 0)
        xlim = kwargs.get('xlim', 0)
        ylim = kwargs.get('ylim', 0)
        title = kwargs.get('title', 'FFT result')
        
        n = len(data)
        k = np.arange(n)
        T = float(n)/fs
        freq = k/T 
        freq = freq[range(int(n/2))]
        
        fig = plt.figure(figsize=(12,5))
        FFT_data = fft(inputData)/n 
        FFT_data = FFT_data[range(int(n/2))]
        ax1 = fig.add_subplot(1,2,1)
        if graphStyle == 0:
            ax1.plot(freq, abs(FFT_data), 'r', linestyle=' ', marker='^') 
        else:
            ax1.plot(freq,abs(FFT_data),'r')
        ax1.set_xlabel('Freq (Hz)')
        ax1.set_ylabel('|Y(freq)|')
        ax1.vlines(freq, [0], abs(FFT_data))
        ax1.set_title('input : ' + title)
        
        ax2 = fig.add_subplot(1,2,2)
        FFT_data = fft(data)/n 
        FFT_data = FFT_data[range(int(n/2))]
        if graphStyle == 0:
            ax2.plot(freq, abs(FFT_data), 'r', linestyle=' ', marker='^') 
        else:
            ax2.plot(freq,abs(FFT_data),'r')
        ax2.set_xlabel('Freq (Hz)')
        ax2.set_ylabel('|Y(freq)|')
        ax2.vlines(freq, [0], abs(FFT_data))
        ax2.set_title('LPF : ' + title)

        plt.title('FFT : ' + title)
        plt.grid(True)
        plt.savefig(title + '_FFT')
        
        
    def analyze(self):
        num = np.array([1.])
        den = np.array([self.tau, 1.])
        w, h = sig.freqs(num, den, worN=np.logspace(0, 5, 1000))

        # Design 1st Order Low Pass Filter Z-Transform
        num_z = np.array([self.Ts/(self.tau + self.Ts)])
        den_z = np.array([1., -self.tau/(self.tau+self.Ts)])

        wz, hz = sig.freqz(num_z, den_z, worN=100000)
        
        plt.figure(figsize=(12,5))
        plt.semilogx(self.toHz(wz*self.Fs), 20 * np.log10(abs(hz)), 'r', label='discrete time')
        plt.semilogx(self.toHz(w), 20 * np.log10(abs(h)), 'b--', label='continuous time')
        plt.axvline(self.f_cut, color='k', lw=1)
        plt.axvline(self.Fs/2, color='k', lw=1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude response [dB]')
        tmpTitle = 'Board LPF : ' + self.title + ', cutoff : ' + str(self.f_cut)
        plt.title(tmpTitle)
        #plt.xlim(1, Fs/2)
        plt.grid()
        plt.legend()
        plt.savefig(self.title + '_boardPlot')
        
        
        # Implementation Signal
        a1 = den_z[1]
        a2 = 0
        b0 = num_z[0]
        b1 = 0.
        b2 = 0.
        
        self.draw_FFT_Graph(self.orig_data[:,0], self.prev_data[:,0], self.Fs, title=self.title, xlim=(0, 500))                



if __name__ == "__main__":
    filter = filter(f_cut=5000, freq=20000)
    data = [100, 100, 100, 100, 100, 100, 200, 200, 200]
    for _data in data:
        print(filter.LPF(
            np.array([float(_data)])
            ))
    filter.plot()
    filter.analyze()