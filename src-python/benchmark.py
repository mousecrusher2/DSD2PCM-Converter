import os
import time
from argparse import ArgumentParser
from pathlib import Path

from app.model.converter import ConversionSettings, convert_dsf_to_flac


def measure_func_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    return result, elapsed


def main():
    parser = ArgumentParser(description="Benchmark DSP functions")
    parser.add_argument("file", help="Input file for benchmarking")
    parser.add_argument("outdir", help="Output file for benchmarking")
    parser.add_argument("--samplerate", type=int, default=88200, help="PCM sample rate")
    parser.add_argument(
        "--stopband", type=float, default=22000.0, help="Stopband frequency in Hz"
    )
    parser.add_argument(
        "--attenuation", type=float, default=145.0, help="Stopband attenuation in dB"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Maximum number of worker threads",
    )
    args = parser.parse_args()
    input_file = args.file
    setting = ConversionSettings(
        output_dir=Path(args.outdir),
        pcm_samplerate=args.samplerate,
        stopband_hz=args.stopband,
        stopband_atten_db=args.attenuation,
        max_workers=args.max_workers,
    )
    result, elapsed = measure_func_time(convert_dsf_to_flac, input_file, setting)
    print(f"Elapsed time: {elapsed:.2f} seconds")
    if result.success:
        print(f"Conversion succeeded: {result.dst_path}")
    else:
        print(f"Conversion failed: {result.message}")


if __name__ == "__main__":
    main()
