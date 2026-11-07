# Leafy Vegetable Size Classifier

**Author:** Renuka Prasad V S

A small classical computer vision project that measures the visible area of one leafy vegetable against a plain contrasting background. It segments green pixels in HSV, cleans the mask with 5×5 morphological closing and opening, selects the largest contour, and classifies the result as **Small** or **Large**.

## Requirements

- Python 3.10 or newer
- OpenCV and NumPy
- A webcam for live mode (optional)

Install dependencies in a virtual environment:

```bash
cd Leaf-Dimension
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Calibration

Keep the camera at a fixed distance and angle with even lighting. Put a coin or other object with known width in the same plane as the leaf. Measure its width in pixels in the captured image, then pass that number as `--reference-pixels`. The default known width is 2.5 cm; change it with `--reference-cm` when using another reference.

The pixel scale is `reference_pixels / reference_cm`. The leaf contour area is divided by the square of that scale to estimate cm². Set the Small/Large boundary with `--threshold`; the default is 100 cm². Choose a threshold based on measured examples from your intended vegetables and camera setup.

## Use

Classify a single image and save an annotated copy beside it:

```bash
python leafy_classifier.py --image path/to/leaf.jpg --reference-pixels 120 --reference-cm 2.5 --threshold 100
```

Choose an output path explicitly:

```bash
python leafy_classifier.py --image path/to/leaf.jpg --reference-pixels 120 --output output/leaf_result.jpg
```

Run live camera mode; press `q` to quit:

```bash
python leafy_classifier.py --camera 0 --reference-pixels 120 --reference-cm 2.5 --threshold 100
```

Classify every supported image in a folder. Annotated files and `results.csv` are written to a `classified` subfolder by default:

```bash
python leafy_classifier.py --batch path/to/images --reference-pixels 120 --threshold 100
```

Adjust the green HSV range if the lighting or leaf color needs it. OpenCV HSV uses H from 0 to 179 and S/V from 0 to 255:

```bash
python leafy_classifier.py --image leaf.jpg --reference-pixels 120 --lower-hsv 25,40,30 --upper-hsv 95,255,255
```

If no reference pixel width is supplied, the app reports pixel² and `Uncalibrated`; it cannot report cm² or Small/Large without calibration.

## YOLO segmentation

`leafy_yolo.py` provides a YOLO instance-segmentation alternative. It measures the largest predicted object mask and applies the same calibration and Small/Large threshold as the HSV script. Install the additional dependency with `python3 -m pip install -r requirements.txt`.

Run it on an image:

```bash
python3 leafy_yolo.py --image path/to/leaf.jpg --reference-pixels 120 --reference-cm 2.5 --threshold 100
```

The default `yolo26n-seg.pt` is a general-purpose model and may not detect an isolated leaf. For reliable leaf masks, pass segmentation weights trained on labeled leaf images with `--model path/to/leaf-seg.pt`. The model needs instance mask annotations for training; a standard image-classification model cannot provide the masks needed for area measurement. Batch and camera modes are also available with `--batch` and `--camera`.

## Method and limitations

The program uses no trained model. Green HSV thresholding works best with a plain white or black background and can be affected by shadows, uneven light, or green backgrounds. It measures the 2D visible contour area, not volume. Touching or overlapping leaves can merge into one contour. The largest green contour is treated as the vegetable, so remove other green objects from the frame.
