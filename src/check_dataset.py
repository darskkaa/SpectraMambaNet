import argparse
import sys
from PIL import Image

def check_dataset(data_dir):
    print("====================================================")

    print(f"🔍 Loading Official WACV Polyp Benchmark TrainDataset from {data_dir}...")
    try:
        from data.datamodule import CustomPolypDataset
        dataset = CustomPolypDataset(data_dir)
    except Exception as e:
        print(f"❌ Failed to load dataset. Error: {e}")
        sys.exit(1)

    total_samples = len(dataset)
    print(f"✅ Loaded train dataset successfully. Total samples: {total_samples}")
    print("\n🧐 Verifying images and masks integrity...")

    corrupted_count = 0
    mismatched_sizes = 0
    valid_count = 0

    # Check a subset or check all depending on user requirements. Checking all is fast.
    for i in range(total_samples):
        try:
            # Read raw PIL sizes directly — CustomPolypDataset.__getitem__ force-resizes
            # the mask to the image's dimensions before returning, so comparing its
            # output shapes is always a match and would never catch a real mismatch.
            img_size = Image.open(dataset.image_paths[i]).size
            mask_size = Image.open(dataset.mask_paths[i]).size

            if img_size != mask_size:
                mismatched_sizes += 1
                print(f"⚠️ Index {i}: Size mismatch between Image {img_size} and Mask {mask_size} "
                      f"({dataset.image_paths[i]} vs {dataset.mask_paths[i]})")
            else:
                valid_count += 1

        except Exception as e:
            corrupted_count += 1
            print(f"❌ Index {i}: Corrupted sample detected! Error: {e}")

    print("\n================= Verification Summary =================")
    print(f"Total Checked:      {total_samples}")
    print(f"Successfully Valid: {valid_count}")
    print(f"Size Mismatches:    {mismatched_sizes}")
    print(f"Corrupted Files:    {corrupted_count}")
    
    if corrupted_count == 0 and mismatched_sizes == 0:
        print("\n🚀 All files verified! Your local cache is 100% healthy.")
    else:
        print("\n⚠️ Issues detected — re-download the dataset before training.")
    print("========================================================")
    return corrupted_count == 0 and mismatched_sizes == 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/content/TrainDataset",
                    help="merged PraNet TrainDataset root (locally: TrainDataset/TrainDataset)")
    args = ap.parse_args()
    sys.exit(0 if check_dataset(args.data_dir) else 1)
