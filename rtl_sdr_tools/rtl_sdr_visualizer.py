"""
RTL-SDR V4 Real-time Spectrum Visualizer
Показывает спектр и waterfall в реальном времени с RTL-SDR V4.

Использование:
    python rtl_sdr_visualizer.py                  # 145.800 MHz (ISS)
    python rtl_sdr_visualizer.py -f 100.0         # 100.0 MHz
    python rtl_sdr_visualizer.py -f 145.8 -g 40   # gain 40 dB
"""

import sys
import time
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

try:
    from rtlsdr import RtlSdr

    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    print("❌ rtlsdr не установлен: pip install pyrtlsdr==0.2.93")
    sys.exit(1)


class RTLSDRVisualizer:
    """Визуализатор спектра RTL-SDR в реальном времени."""

    def __init__(
        self,
        frequency: float = 145.800,
        gain: float = 20.0,
        sample_rate: float = 2.4e6,
        fft_size: int = 4096,
        history: int = 200,
    ):
        self.frequency = frequency  # MHz
        self.gain = gain  # dB
        self.sample_rate = int(sample_rate)
        self.fft_size = fft_size
        self.history = history  # Количество кадров waterfall

        # RTL-SDR
        self.sdr = None
        self._hann_window = np.hanning(fft_size)

        # История для waterfall
        self.waterfall_data = deque(maxlen=history)

        # Настройка графика
        self.fig = plt.figure(figsize=(14, 8), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title(f"RTL-SDR V4 Spectrum — {frequency} MHz")
        plt.style.use("dark_background")

        # Создаём subplot'ы
        gs = self.fig.add_gridspec(3, 1, height_ratios=[2, 2, 0.3])
        self.ax_spectrum = self.fig.add_subplot(gs[0])
        self.ax_waterfall = self.fig.add_subplot(gs[1])
        self.ax_info = self.fig.add_subplot(gs[2])
        self.ax_info.axis("off")

        self._setup_plots()

    def _setup_plots(self):
        """Настройка осей и стилей."""
        # Спектр
        self.ax_spectrum.set_title("Spectrum", fontsize=14, color="white", pad=10)
        self.ax_spectrum.set_ylabel("Power (dB)", fontsize=10, color="white")
        self.ax_spectrum.tick_params(colors="white")
        self.ax_spectrum.grid(True, alpha=0.3)
        self.ax_spectrum.set_facecolor("#16213e")

        # X-axis: частота в MHz
        freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1.0 / self.sample_rate))
        self.freq_offset = freqs  # Hz
        self.freq_mhz = (freqs / 1e6) + self.frequency  # MHz

        (self.line_spectrum,) = self.ax_spectrum.plot(
            self.freq_mhz, np.zeros_like(freqs), color="#00ff88", linewidth=0.8, alpha=0.9
        )

        xlim = (self.frequency - self.sample_rate / 2e6, self.frequency + self.sample_rate / 2e6)
        self.ax_spectrum.set_xlim(xlim)

        # Waterfall
        self.ax_waterfall.set_title("Waterfall", fontsize=14, color="white", pad=10)
        self.ax_waterfall.set_ylabel("Time", fontsize=10, color="white")
        self.ax_waterfall.set_xlabel("Frequency (MHz)", fontsize=10, color="white")
        self.ax_waterfall.tick_params(colors="white")
        self.ax_waterfall.set_facecolor("#16213e")

        # Инициализация waterfall
        self.waterfall_img = self.ax_waterfall.imshow(
            np.zeros((self.history, self.fft_size)),
            aspect="auto",
            cmap="viridis",
            extent=[xlim[0], xlim[1], self.history, 0],
            vmin=-80,
            vmax=-20,
        )

        # Info text
        self.info_text = self.ax_info.text(
            0.02,
            0.5,
            "",
            transform=self.ax_info.transAxes,
            fontsize=10,
            color="#00ff88",
            family="monospace",
        )

    def capture_spectrum(self) -> tuple:
        """Захват одного кадра спектра."""
        try:
            samples = self.sdr.read_samples(self.fft_size)
            if samples is None or len(samples) == 0:
                return None, None

            # Hann window + FFT
            windowed = samples * self._hann_window
            fft_result = np.fft.fftshift(np.fft.fft(windowed))
            power_db = 10 * np.log10(np.abs(fft_result) ** 2 + 1e-12)

            # Нормализация для лучшего отображения
            power_db = np.clip(power_db, -80, -20)

            return self.freq_mhz, power_db

        except Exception as e:
            print(f"⚠️  Ошибка захвата: {e}")
            return None, None

    def init_animation(self):
        """Инициализация анимации."""
        self.line_spectrum.set_ydata(np.zeros(self.fft_size))
        return self.line_spectrum, self.waterfall_img, self.info_text

    def update_animation(self, frame):
        """Обновление кадра анимации."""
        # Захватываем спектр
        freq_mhz, power_db = self.capture_spectrum()

        if power_db is None:
            return self.line_spectrum, self.waterfall_img, self.info_text

        # Обновляем спектр
        self.line_spectrum.set_ydata(power_db)

        # Автомасштаб Y
        self.ax_spectrum.set_ylim(np.min(power_db) - 5, np.max(power_db) + 5)

        # Добавляем в waterfall
        self.waterfall_data.append(power_db)

        # Обновляем waterfall
        waterfall_array = np.array(self.waterfall_data)
        self.waterfall_img.set_array(waterfall_array)

        # Info
        signal_strength = np.mean(power_db)
        peak_freq = freq_mhz[np.argmax(power_db)]
        timestamp = time.strftime("%H:%M:%S")

        self.info_text.set_text(
            f"  Time: {timestamp}  |  "
            f"Freq: {self.frequency} MHz  |  "
            f"SR: {self.sample_rate / 1e6:.1f} MSPS  |  "
            f"Gain: {self.gain} dB  |  "
            f"Signal: {signal_strength:.1f} dB  |  "
            f"Peak: {peak_freq:.3f} MHz  |  "
            f"Frames: {len(self.waterfall_data)}"
        )

        return self.line_spectrum, self.waterfall_img, self.info_text

    def initialize_sdr(self) -> bool:
        """Инициализация RTL-SDR."""
        print(f"\n🔌 Инициализация RTL-SDR V4...")
        print(f"   Частота: {self.frequency} MHz")
        print(f"   Sample rate: {self.sample_rate / 1e6:.1f} MSPS")
        print(f"   Gain: {self.gain} dB\n")

        try:
            self.sdr = RtlSdr()
            self.sdr.rs = self.sample_rate
            self.sdr.fc = int(self.frequency * 1e6)
            self.sdr.gain = self.gain

            tuner = self.sdr.get_tuner_type()
            print(f"✅ RTL-SDR подключен (Tuner: {tuner})")
            return True

        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}")
            return False

    def run(self):
        """Запуск визуализации."""
        if not self.initialize_sdr():
            sys.exit(1)

        print("📊 Запуск визуализации...")
        print("   Закройте окно для остановки\n")

        try:
            ani = animation.FuncAnimation(
                self.fig,
                self.update_animation,
                init_func=self.init_animation,
                interval=100,  # 10 FPS
                blit=False,
                cache_frame_data=False,
            )
            plt.tight_layout()
            plt.show()

        except KeyboardInterrupt:
            print("\n⏹️  Остановка по Ctrl+C")
        finally:
            self.cleanup()

    def cleanup(self):
        """Очистка ресурсов."""
        if self.sdr:
            try:
                self.sdr.close()
                print("🔌 RTL-SDR отключен")
            except Exception:
                pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RTL-SDR V4 Spectrum Visualizer")
    parser.add_argument(
        "-f",
        "--frequency",
        type=float,
        default=145.800,
        help="Частота в MHz (default: 145.800 ISS)",
    )
    parser.add_argument(
        "-g", "--gain", type=float, default=20.0, help="Усиление в dB (default: 20)"
    )
    parser.add_argument(
        "-s", "--sample-rate", type=float, default=2.4e6, help="Sample rate в Hz (default: 2.4e6)"
    )
    parser.add_argument("--fft-size", type=int, default=4096, help="FFT size (default: 4096)")

    args = parser.parse_args()

    visualizer = RTLSDRVisualizer(
        frequency=args.frequency,
        gain=args.gain,
        sample_rate=args.sample_rate,
        fft_size=args.fft_size,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
