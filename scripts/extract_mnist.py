import gzip, shutil, os

DATA_DIR = "data/MNIST" # path to MNIST gz files

# list of gz files
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

os.makedirs(DATA_DIR, exist_ok=True)

for f in FILES:
    gz_path = os.path.join(DATA_DIR, f)
    out_path = gz_path[:-3]  # remove .gz extension
    if not os.path.exists(out_path):
        print(f"Extracting {f}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print(f"{f} already extracted.")