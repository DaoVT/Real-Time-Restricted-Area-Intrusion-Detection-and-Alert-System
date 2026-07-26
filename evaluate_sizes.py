import json
from ultralytics import YOLO
import glob

def main():
    print("Loading model best.pt...")
    model = YOLO('best.pt')
    
    data_yaml_path = 'test_and_model/kaggle/input/notebooks/minhnguyncng19072005/notebook560d537205/mega_dataset/data.yaml'
    
    print("Running validation on test split...")
    metrics = model.val(data=data_yaml_path, split='test', save_json=True)
    
    print("Parsing predictions.json...")
    # The output dir might be different each time, so we find the latest predictions.json
    pred_files = glob.glob('runs/detect/val*/predictions.json')
    if not pred_files:
        print("Could not find predictions.json!")
        return
    latest_pred_file = max(pred_files, key=lambda x: int(x.split('val')[-1].replace('\\', '').replace('/', '') or '1'))
    print(f"Reading from {latest_pred_file}")

    with open(latest_pred_file) as f:
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
    
    print("Running prediction on VIRAT (KAGGLE) images for visual inspection...")
    virat_images = glob.glob('test_and_model/kaggle/input/notebooks/minhnguyncng19072005/notebook560d537205/mega_dataset/images/test/KAGGLE_*.jpg')[:5]
    if virat_images:
        model.predict(source=virat_images, save=True, conf=0.15)
        print("Visual predictions saved to runs/detect/predict/")
    else:
        print("No KAGGLE (VIRAT) images found.")

if __name__ == '__main__':
    main()
