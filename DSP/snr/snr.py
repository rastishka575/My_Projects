import numpy as np
import scipy.io.wavfile as sw
import matplotlib.pyplot as plt


def quantization(sig, quant):
    signal_min = abs(sig.min())
    signal = sig + signal_min
    quantization_step = (signal.max() - signal.min())/quant
    signal_quant = np.ceil(signal / quantization_step)*quantization_step
    signal_quant -= signal_min
    return signal_quant


def snr(signal, signal_quant):
    s = 10 * np.log10(np.var(signal)/np.var(signal_quant))
    return s


if __name__ == "__main__":
    time = 1
    Fs = 1000
    f = 10
    path = 'Lab1//Speech//voice.wav'
    n = 10
    quantization_num = 2**n

    # 1 signal
    t = np.linspace(start=0, stop=time, num=time*Fs)
    sig = 2*np.pi*f*t
    sin_signal = np.sin(sig)

    # 2 signal
    random_signal = np.random.uniform(-1, 1, time*Fs)

    # 3 signal
    fs, voice_signal = sw.read(path)
    voice_signal = voice_signal[:time*fs]

    '''
    sw.write('voice_fs1000.wav', Fs, voice_signal)
    _, voice_signal = sw.read('voice_fs1000.wav')
    voice_signal = voice_signal[:time * Fs]
    '''

    # квантование
    sin_signal_quant = quantization(sin_signal, quantization_num)
    random_signal_quant = quantization(random_signal, quantization_num)
    voice_signal_quant = quantization(voice_signal, quantization_num)

    # шум квантования
    sin_signal_noise_quant = abs(sin_signal - sin_signal_quant)
    random_signal_noise_quant = abs(random_signal - random_signal_quant)
    voice_signal_noise_quant = abs(voice_signal - voice_signal_quant)

    # snr
    sin_signal_snr = snr(sin_signal, sin_signal_noise_quant)
    random_signal_snr = snr(random_signal, random_signal_noise_quant)
    voice_signal_snr = snr(voice_signal, voice_signal_noise_quant)

    print('theoretical_snr', 52.8)
    print('sin_signal_snr', sin_signal_snr)
    print('random_signal_snr', random_signal_snr)
    print('voice_signal_snr', voice_signal_snr)

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].hist(sin_signal, bins=Fs)
    axs[0, 0].set_title('sin_signal')
    axs[0, 1].hist(sin_signal_noise_quant, bins=Fs)
    axs[0, 1].set_title('sin_signal_noise_quant')

    axs[1, 0].hist(random_signal, bins=Fs)
    axs[1, 0].set_title('random_signal')
    axs[1, 1].hist(random_signal_noise_quant, bins=Fs)
    axs[1, 1].set_title('random_signal_noise_quant')

    axs[2, 0].hist(voice_signal, bins=Fs)
    axs[2, 0].set_title('voice_signal')
    axs[2, 1].hist(voice_signal_noise_quant, bins=Fs)
    axs[2, 1].set_title('voice_signal_noise_quant')

    plt.show()
