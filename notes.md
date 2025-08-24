### EAST Network Architrecture
- pretrained neural network, so it just spits out an output
- `(scores, geometry) = net.forward(layerNames)` run after image is scaled down
    - images are scaled to consistent resolution
    - EAST divides the image further into consistent chunks
    - each chunk will contain
        1)  `score`, representing the confidence that there is text in that chunk
        2) `geometry`, representing boundaries of text within the chunk. `geometry` is a 4D tensor with five channels, representing distance from top, right, bottom, left, and the rotation



### `geometry` tensor: `[batch_size, channel, height, width]`
```
top_dist    = geometry[0, 0, y, x]  # Distance to top edge
right_dist  = geometry[0, 1, y, x]  # Distance to right edge  
bottom_dist = geometry[0, 2, y, x]  # Distance to bottom edge
left_dist   = geometry[0, 3, y, x]  # Distance to left edge
angle       = geometry[0, 4, y, x]  # Rotation angle
```

- `geometry` predicts text always, regardless of if there is actually text there, but this is still fine since we would only use this information if the confidence score is high for that chunk.
- `geometry` is useful for calculating bounding box dimensions, where the height is just sum of top and bottom distance and width is sum of right and left distance

Essentially, each chunk is saying "I'm (score)% sure text is here and the text region is (geometry[0]) pixels up from top left of chunk, etc. 

### Text Recognition
- loop through all the geometry and check confidence score; if above a minimum threshold, perform calculations to map back onto original image (EAST shrinks the image by a 4x scale)
- Remove overlap with `boxes = non_max_suppression(np.array(rects), probs=confidences)`

Now, there's an array `boxes`, containing the bounding box coordinates.
```
# Extract the Region of Interest (ROI) from original image
roi = orig[startY:endY, startX:endX]

# Configure Tesseract OCR parameters
config = ("-l eng --oem 1 --psm 7")  # English, LSTM engine, single text line

# Apply OCR to extract text from ROI
text = pytesseract.image_to_string(roi, config=config)
```
- Go through every box and use Tesseract on the region of interest and extract the text.