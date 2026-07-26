import json
import glob
from ultralytics import YOLO

def main():
    print("Parsing predictions.json...")
    pred_file = 'runs/detect/val-10/predictions.json'
    print(f"Reading from {pred_file}")

    with open(pred_file) as f:
        preds = json.load(f)

    small, medium, large = [], [], []
    for p in preds:
        area = p['bbox'][2] * p['bbox'][3]  # width * height
        if area < 32*32:
            small.append(p)
        elif area < 96*96:
            medium.append(p)
        else:
            large.append(p)

    print(f"Total predictions: {len(preds)}")
    print(f"Small (< 32x32): {len(small)}")
    print(f"Medium (< 96x96): {len(medium)}")
    print(f"Large (>= 96x96): {len(large)}")
    
    print("\nRunning prediction on VIRAT (KAGGLE) images for visual inspection...")
    model = YOLO('best.pt')
    virat_images = glob.glob('test_and_model/kaggle/input/notebooks/minhnguyncng19072005/notebook560d537205/mega_dataset/images/test/KAGGLE_*.jpg')[:5]
    if virat_images:
        model.predict(source=virat_images, save=True, conf=0.15)
        print("Visual predictions saved to runs/detect/predict/")
    else:
        print("No KAGGLE (VIRAT) images found.")

if __name__ == '__main__':
    main()
