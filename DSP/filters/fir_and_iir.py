import numpy as np
import scipy.signal as sgn
import matplotlib.pyplot as plt


# sinusoidal signal
def sin_signal(f, time=1, Fs=200):
    t = np.linspace(start=0, stop=time, num=time * Fs)
    sig = 2 * np.pi * f * t
    sin_signal = np.sin(sig)
    return sin_signal


# графики передаточной функции
def show_filter(b, title):
    w, h = sgn.freqz(b)

    plt.plot(w / np.pi, abs(h))
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    time = 1
    Fs = 200
    f_list = np.asarray([15, 30, 50, 75, 90])
    sig_list = dict()

    # signal with frequency 15, 30, 50, 75, 90 HZ
    signal = None
    for f in f_list:
        if signal is None:
            signal = sin_signal(f)
        else:
            signal += sin_signal(f)

    sig_list.update({'signal': signal})

    freq_fir = f_list/(0.5*Fs*time)

    # IIR
    order = 6

    b0, a = sgn.iirfilter(order, [freq_fir[1] + 0.15, freq_fir[3]-0.15], btype='bandstop')
    signal = sgn.filtfilt(b0, a, signal)

    show_filter(b0, 'IIR_filter')

    # FIR
    order = 65

    b1 = sgn.firwin(order, freq_fir[0] + 0.05, pass_zero=False)
    signal = sgn.filtfilt(b1, 1, signal)

    b2 = sgn.firwin(order, freq_fir[-1] - 0.05, pass_zero=True)
    signal = sgn.filtfilt(b2, 1, signal)

    show_filter(b1+b2, 'FIR_filter')

    sig_list.update({'signal_update': signal})

    # visualization spectrum
    plt.plot(np.abs(np.fft.fft(sig_list['signal'])[:len(sig_list['signal']) // 2]), label='signal')
    plt.plot(np.abs(np.fft.fft(sig_list['signal_update'])[:len(sig_list['signal_update']) // 2]), label='signal_update')
    plt.legend()
    plt.title('spectrum')
    plt.show()

    # visualization signal
    plt.plot(sig_list['signal'], label='signal')
    plt.plot(sig_list['signal_update'], label='signal_update')
    plt.legend()
    plt.title('signal')
    plt.show()

