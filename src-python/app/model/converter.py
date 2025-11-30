from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from threadpoolctl import threadpool_limits

from .dsf_reader import DsfReader
from .dsp import design_kaiser_lowpass, fir_decimate_chunk_stateless
from .tagging import copy_tags_dsf_to_flac


@dataclass(frozen=True)
class ConversionSettings:
    output_dir: Path
    pcm_samplerate: int
    stopband_hz: float
    stopband_atten_db: float
    max_workers: int


@dataclass(frozen=True)
class ConversionResult:
    success: bool
    src_path: Path
    dst_path: Path | None
    message: str


def convert_dsf_to_flac(
    src_path: str | Path,
    settings: ConversionSettings,
    progress_cb: Callable[[float], None] | None = None,
) -> ConversionResult:
    src = Path(src_path)
    try:
        if not src.exists():
            return ConversionResult(False, src, None, "Source file does not exist.")

        out_dir = settings.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / (src.stem + ".flac")
        return convert_inner(src, dst, settings, progress_cb)

    except Exception as exc:
        return ConversionResult(False, src, None, f"Error: {exc}")


def convert_inner(
    src: Path,
    dst: Path,
    settings: ConversionSettings,
    progress_cb: Callable[[float], None] | None = None,
) -> ConversionResult:
    fs_pcm = int(settings.pcm_samplerate)
    with (
        DsfReader(src) as reader,
        sf.SoundFile(
            dst,
            mode="w",
            samplerate=fs_pcm,
            channels=reader.channels,
            format="FLAC",
            subtype="PCM_24",
        ) as out_f,
        ThreadPoolExecutor(max_workers=settings.max_workers) as executor,
        threadpool_limits(limits=1, user_api="blas"),
    ):
        fs_dsd = reader.sample_rate
        channels = reader.channels
        total_samples = reader.sample_count

        if progress_cb is not None:
            progress_cb(0.0)

        if fs_pcm <= 0:
            return ConversionResult(
                False, src, None, "PCM sample rate must be positive."
            )

        if fs_dsd % fs_pcm != 0:
            msg = (
                f"DSD sample rate {fs_dsd} is not an integer multiple of "
                f"target PCM rate {fs_pcm}."
            )
            return ConversionResult(False, src, None, msg)

        decim = fs_dsd // fs_pcm
        print(f"Decimation factor: {decim}")

        # FIR 設計（float32）
        max_stop = 0.45 * fs_pcm
        stopband_hz = min(float(settings.stopband_hz), max_stop)
        if stopband_hz <= 0.0:
            stopband_hz = max_stop

        taps = design_kaiser_lowpass(
            fs=float(fs_dsd),
            f_stop=stopband_hz,
            attenuation_db=float(settings.stopband_atten_db),
        )
        taps = np.ascontiguousarray(taps.astype(np.float64)[::-1])
        L = len(taps)
        print(f"FIR taps length: {L}")
        if L < 2:
            return ConversionResult(False, src, None, "FIR taps length is too short.")

        overlap = L - 1

        chunk_dsd_samples = calc_chunksize(channels, fs_dsd, overlap)

        # 直前までの末尾 overlap サンプル（uint8）
        tail = np.zeros((overlap, channels), dtype=np.uint8)

        # グローバル DSD インデックス
        global_index = 0

        agg_blocks: list[np.ndarray] = []
        agg_count = 0

        # 進捗表示用
        processed_samples = 0  # 変換完了したDSD サンプル数（per channel）

        # チャンク ID と書き出し順管理
        next_chunk_id = 0  # 次に submit するチャンクの ID
        next_write_id = 0  # 次に out_f に書き出すべきチャンク ID
        pending: dict[int, Future[np.ndarray]] = {}

        def submit_chunk(main_tail: np.ndarray, g_start: int) -> None:
            nonlocal next_chunk_id
            chunk_id = next_chunk_id
            next_chunk_id += 1

            dsd_ext = main_tail

            fut: Future[np.ndarray] = executor.submit(
                fir_decimate_chunk_stateless,
                dsd_ext,
                taps,
                decim,
                g_start,
                overlap,
            )
            pending[chunk_id] = fut

        def drain_completed(max_future: int | None = None) -> None:
            """
            chunk_id の昇順で、完了済みのチャンクを out_f に書き出す。

            max_future が指定された場合は、その数以下に抑える。
            つまり、完了済みチャンクがあっても max_future 以下なら
            書き出しを止める。
            なお、max_future=None の場合は完了しているものをすべて書き出す。
            """
            nonlocal next_write_id

            if max_future is None:
                while (fut := pending.get(next_write_id)) and fut.done():
                    pcm_block = fut.result()
                    if pcm_block.size != 0:
                        out_f.write(pcm_block)

                    del pending[next_write_id]
                    next_write_id += 1
            else:
                while (fut := pending.get(next_write_id)) and len(pending) > max_future:
                    pcm_block = fut.result()
                    if pcm_block.size != 0:
                        out_f.write(pcm_block)

                    del pending[next_write_id]
                    next_write_id += 1

        # DSD ブロックを読みながらチャンク分割して、その場で FIR+decimate する
        for blk in reader.iter_blocks():
            if blk.size == 0:
                continue
            if blk.dtype != np.uint8:
                raise ValueError("DSD block dtype must be uint8.")

            # 「読み終わった DSD サンプル数」を加算（進捗表示用）
            processed_samples += blk.shape[0]
            if progress_cb is not None and total_samples > 0:
                frac = min(processed_samples / float(total_samples), 0.9999)
                progress_cb(frac)

            agg_blocks.append(blk)
            agg_count += blk.shape[0]

            # まとめた DSD サンプル数がチャンクしきい値を超えたら処理
            while agg_count >= chunk_dsd_samples:
                # チャンク本体を取り出す
                big = np.concatenate(agg_blocks, axis=0)
                main = big[:chunk_dsd_samples, :]
                rest = big[chunk_dsd_samples:, :]

                agg_blocks = [rest] if rest.size > 0 else []
                agg_count = rest.shape[0] if rest.size > 0 else 0

                concat_for_tail = np.concatenate([tail, main], axis=0)
                # このチャンクをスレッドプールに投げる
                submit_chunk(concat_for_tail, global_index)

                # tail 更新（次チャンク用）: ここは入力 DSD だけで決まるので
                # 計算結果は待たなくてよい
                if concat_for_tail.shape[0] >= overlap:
                    tail = concat_for_tail[-overlap:, :]
                else:
                    pad = overlap - concat_for_tail.shape[0]
                    new_tail = np.zeros((overlap, channels), dtype=np.uint8)
                    new_tail[pad:, :] = concat_for_tail
                    tail = new_tail

                global_index += main.shape[0]

                # 溜めすぎ防止: pending が多くなったら少し捌く
                drain_completed(max_future=2 * settings.max_workers)

        # 余りチャンクがあれば最後に処理（これもスレッドプールに投げる）
        if agg_count > 0:
            big = np.concatenate(agg_blocks, axis=0)
            main = big

            submit_chunk(np.concatenate([tail, main], axis=0), global_index)

        # すべての DSD を処理し終わったので、最終進捗を 1.0 に
        if progress_cb is not None:
            progress_cb(1.0)

        # すべてのチャンクが終わるまで待って順番に書き出す
        drain_completed(max_future=0)
        # （executor は with ブロックを抜けると自動で shutdown(wait=True)）

    # タグコピー
    try:
        copy_tags_dsf_to_flac(src, dst)
    except Exception as tag_exc:
        return ConversionResult(
            True,
            src,
            dst,
            f"Converted, but failed to copy tags: {tag_exc}",
        )

    return ConversionResult(True, src, dst, "OK")


def calc_chunksize(
    channels: int,
    fs_dsd: int,
    overlap: int,
) -> int:
    # -----------------------------
    # チャンクサイズを動的に決める
    #   ・基本は 0.5 秒
    #   ・ただし DSD256 のような高 Fs では、チャンクのバイト数が
    #     32MB を超えないように制限
    # -----------------------------
    target_chunk_bytes = 32 * 1024 * 1024  # 32MB くらいを目安
    bytes_per_sample = 4 * channels  # float32 × channels

    chunk_dsd_samples_time = int(fs_dsd * 0.5)  # 0.5 秒
    chunk_dsd_samples_mem = target_chunk_bytes // bytes_per_sample

    chunk_dsd_samples = min(chunk_dsd_samples_time, chunk_dsd_samples_mem)
    if chunk_dsd_samples <= overlap:
        chunk_dsd_samples = overlap * 2
    return chunk_dsd_samples
