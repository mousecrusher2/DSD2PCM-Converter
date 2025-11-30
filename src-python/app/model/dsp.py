from __future__ import annotations

import fir_decimator
import numpy as np


def design_kaiser_lowpass(
    fs: float,
    f_stop: float,
    attenuation_db: float,
    transition_ratio: float = 0.1,
) -> np.ndarray:
    """Kaiser 窓でローパス FIR フィルタを設計する。

    Parameters
    ----------
    fs:
        入力サンプリング周波数 [Hz] (DSD 側の Fs)。
    f_stop:
        阻止帯域開始周波数 [Hz]。
    attenuation_db:
        阻止帯域減衰量 [dB]。
    transition_ratio:
        遷移帯域幅を f_stop に対する比 (0.1 なら f_stop の ±10% を遷移帯域にするイメージ)。
    """
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")
    if f_stop <= 0:
        raise ValueError("Stopband frequency must be positive.")

    # Nyquist 未満にクランプ
    nyq = fs / 2.0
    f_stop = min(f_stop, 0.99 * nyq)

    # 遷移帯域幅
    delta_f = max(f_stop * transition_ratio, fs * 1e-5)
    if delta_f <= 0:
        raise ValueError("Transition width must be positive.")

    A = max(float(attenuation_db), 0.0)

    # Kaiser β
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21.0) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0

    # 正規化遷移幅 Δω [rad]
    dw = 2.0 * np.pi * delta_f / fs
    # フィルタ長の近似 (Oppenheim & Schafer)
    N = int(np.ceil((A - 8.0) / (2.285 * dw)))
    if N < 5:
        N = 5
    # 奇数長にして線形位相 FIR にする
    if N % 2 == 0:
        N += 1

    # 遷移帯域の中央をカットオフに設定
    fc = max(min(f_stop - delta_f / 2.0, nyq * 0.99), 1.0)

    n = np.arange(N)
    m = n - (N - 1) / 2.0

    # 理想 LPF: 2*fc/fs * sinc(2*fc*m/fs)
    x = 2.0 * fc * m / fs
    h_ideal = 2.0 * fc / fs * np.sinc(x)

    # Kaiser 窓
    if beta == 0.0:
        window = np.ones_like(h_ideal)
    else:
        arg = beta * np.sqrt(1.0 - ((2.0 * n) / (N - 1) - 1.0) ** 2)
        window = np.i0(arg) / np.i0(beta)

    h = h_ideal * window
    # DC 利得 = 1 に正規化
    h /= np.sum(h)
    return h.astype(np.float32)

def fir_decimate_chunk_stateless(
    dsd_ext: np.ndarray,
    taps_reversed: np.ndarray,
    decim: int,
    global_start_index: int,  # 本体 main[0] のグローバルインデックス
    overlap: int,
) -> np.ndarray:
    """stateless な 1 チャンク用 FIR+decimation（float32版）。

    Parameters
    ----------
    dsd_ext:
        shape = (N_ext, ch)。先頭 overlap サンプルはオーバーラップ部分、
        続く main_len サンプルが今回のチャンク本体。
    taps:
        FIR 係数 (1D)。
    decim:
        decimation factor。
    global_start_index:
        本体 main[0] のグローバルインデックス（dsd 全体に対して）。
    overlap:
        オーバーラップサンプル数（通常 taps.size - 1）。
    """
    # 型とメモリレイアウトを Cython 側に合わせる
    dsd_ext32 = dsd_ext
    # print(dsd_ext32.dtype, dsd_ext32.flags, dsd_ext32.shape)
    taps64 = taps_reversed

    num_samples, num_channels = dsd_ext32.shape

    if overlap < 0 or overlap > num_samples:
        raise ValueError("invalid overlap")

    # 本体サンプル数 = 全体 - overlap
    main_len = num_samples - overlap
    if main_len <= 0:
        # 本体がない場合は空配列
        return np.empty((0, num_channels), dtype=np.float64)

    # dsd_ext[0] のグローバルインデックス
    global_start_ext = global_start_index - overlap

    # Cython コアに渡す位相 (0..decim-1)
    phase_init = int(global_start_ext % decim)

    # Cython 実装を呼び出し
    pcm = fir_decimator.fir_decimate_chunk_core(
        dsd_ext32,
        taps64,
        int(decim),
        phase_init,
        int(overlap),
        int(main_len),
    )

    # _fir_decimate_chunk_core は float64 を返す実装にしている想定
    return pcm
