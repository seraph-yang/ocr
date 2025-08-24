# OpenCV OCR

Performs text detection + text recorgnition using OpenCV, Python, and Tesseract

OCR utilizees an LSTM, a type of RNN

1) Use OpenCV EAST text detector to find text in an image -> gives bounding box coordinates
2) Extract each of these Region of Interests (ROI) and pass into Tessearct deep learning recognition algorithm
3) Output gives us OCR results
4) Draw results on output

Tesseract binary needs flags:
- `-l`: language of input text
- `-oem`: OCR Engine Mode, controls algorithm used
- `psm`: Page Segmentation Mode

### Installations
```
pip install opencv-python numpy pytesseract imutils
```

### Run
```
source ocr_env/bin/activate
python text_recognition.py --east frozen_east_text_detection.pb --image images/image1.png
```

Adding `--padding 0.25` increases the bounding box; larger number means bigger bounding box