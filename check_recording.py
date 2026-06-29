import pickle
import os

total_frames = 0
files = [f for f in os.listdir('.') if f.startswith('recording_') and f.endswith('.pkl')]
files.sort()

for file in files:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    duration = data[-1]['timestamp'] - data[0]['timestamp']
    click_frames = sum(1 for d in data if d['clicking'])
    print(f"{file}: {len(data)} frames | {duration:.0f}s | clicking {click_frames/len(data):.0%} of time")
    total_frames += len(data)

print(f"\nTotal frames across all recordings: {total_frames}")
print(f"Total recordings: {len(files)}")