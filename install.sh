#!/bin/bash
pip install --no-cache-dir --no-deps --force-reinstall git+https://github.com/unslothai/unsloth.git
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt
