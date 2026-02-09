import os
os.environ['TRANSFORMERS_NO_TF'] = '1'  # Force PyTorch only, disable TensorFlow

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
