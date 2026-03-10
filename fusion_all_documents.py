from utils import load_object_from_disk, save_object_to_disk
from tqdm import tqdm
from glob import glob
import os

all_documents: dict[str, str] = {}

mcts_output_dir = "mcts_output"
fps = glob(os.path.join(mcts_output_dir, "**", "documents.pkl.zst"), recursive=True)
for fp in tqdm(fps, ncols=80, desc="Loading documents"):
    try:
        documents = load_object_from_disk(fp)
        all_documents.update(documents)
    except Exception as e:
        print(f"Error loading {fp}: {e}")
print(f"Total documents loaded: {len(all_documents)}")
save_object_to_disk(
    all_documents,
    os.path.join(mcts_output_dir, "all_documents.pkl.zst"),
    compress_level=20,
)
print(
    f"Saved all documents to {os.path.join(mcts_output_dir, 'all_documents.pkl.zst')}"
)
