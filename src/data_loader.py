"""FakeAVCeleb dataset utilities (work in progress)."""

import os

import cv2

# category folder -> (binary label, category label)
# binary: 0 = real, 1 = fake | category: 0 RVRA, 1 FVRA, 2 RVFA, 3 FVFA
CATEGORIES = {
    "RealVideo-RealAudio": (0, 0),
    "FakeVideo-RealAudio": (1, 1),
    "RealVideo-FakeAudio": (1, 2),
    "FakeVideo-FakeAudio": (1, 3),
}

IMG_SIZE = 224
FRAMES_PER_CLIP = 8


def list_videos(root):
    """Walk the dataset folder, return (path, binary_label, category_label)."""
    samples = []
    for cat_name, (binary, category) in CATEGORIES.items():
        cat_dir = os.path.join(root, cat_name)
        if not os.path.isdir(cat_dir):
            continue
        for speaker in sorted(os.listdir(cat_dir)):
            speaker_dir = os.path.join(cat_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            for fname in sorted(os.listdir(speaker_dir)):
                if fname.lower().endswith((".mp4", ".avi")):
                    samples.append((os.path.join(speaker_dir, fname), binary, category))
    return samples


def extract_frames(video_path, num_frames=FRAMES_PER_CLIP):
    """Grab evenly spaced frames from a clip. Face cropping still TODO."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = [int(i * total / num_frames) for i in range(num_frames)]
    frames = []
    pos = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if pos in idxs:
            frames.append(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)))
        pos += 1
    cap.release()
    return frames


if __name__ == "__main__":
    root = os.path.join("data", "FakeAVCeleb_v1.2")
    clips = list_videos(root)
    print("found", len(clips), "videos")
    if clips:
        print(clips[0])
