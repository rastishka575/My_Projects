import numpy as np
import scipy.io.wavfile as sw
import matplotlib.pyplot as plt


def calculate_spectrogram(signal, start_signal, window, length_frame, length_buffer):
    buffer = []         # буффер
    spectrogram = []    # спектрограмма
    f0 = []             # f0

    # проход окна по спектрограмме
    for start_index in range(len(signal[start_signal:start_signal + length_sample])):
        frame = signal[start_signal + start_index:start_signal + start_index + length_frame]

        frame = frame * window  # умножаем на окно Ханна

        # вычисляем спектр участка
        spectrum = np.fft.fft(frame)
        spectrum = spectrum[:length_frame // 2]
        spectrum = np.abs(spectrum)

        buffer.append(spectrum)

        if len(buffer) == length_buffer:
            buffer = np.asarray(buffer)
            spectrum_mean = buffer.mean(axis=0)
            spectrogram.append(spectrum_mean)
            f0.append(spectrum_mean.argmax())
            buffer = []

    f0 = np.asarray(f0)
    spectrogram = np.asarray(spectrogram)

    # логарифмическая спектрограмма
    spectrogram = 10*np.log10(spectrogram)

    spectrogram /= spectrogram.max()
    spectrogram *= 255
    spectrogram = spectrogram.astype(dtype='int')

    spectrogram = np.transpose(spectrogram)
    spectrogram = spectrogram[::-1, :]

    return spectrogram, f0


if __name__ == "__main__":

    path = 'voice.wav'

    # считываем сигнал
    fs, signal = sw.read(path)

    step = fs//1000                     # длина 1 мс (16)
    length_frame = 16*step              # длина участка (16 мс)
    window = np.hanning(length_frame)   # окно Ханна
    length_buffer = 3*step              # размер буффера (3 мс)
    length_sample = fs//2               # временной отрезок для спектрограммы (1 секунда)

    start_voice = fs * 3                # голос
    start_quiet = fs * 4                # тишина

    spectrogram_voice, f0_voice = calculate_spectrogram(signal, start_voice, window, length_frame, length_buffer)
    spectrogram_quiet, f0_quiet = calculate_spectrogram(signal, start_quiet, window, length_frame, length_buffer)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(spectrogram_voice)
    axs[0, 0].set_title('Spectrogram Voice')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Frequency')

    axs[0, 1].plot(f0_voice)
    axs[0, 1].set_title('f0 Voice')
    axs[0, 1].set_xlabel('Time')

    axs[1, 0].imshow(spectrogram_quiet)
    axs[1, 0].set_title('Spectrogram Silence')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Frequency')

    axs[1, 1].plot(f0_quiet)
    axs[1, 1].set_title('f0 Silence')
    axs[1, 1].set_xlabel('Time')

    plt.show()
