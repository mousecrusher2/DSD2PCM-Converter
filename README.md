# DSD2PCM-Converter

DSD2PCM-Converter

cythonからrustにして3倍くらい速くなった。

ライブラリのインストール

```pwsh
uv sync --frozen
.\.venv\Scripts\activate
```

rustビルド

```pwsh
maturin develop --release
```
