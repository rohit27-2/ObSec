from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create the uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class CropLayer(object):
    def __init__(self, params, blobs):
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.startY:self.endY, self.startX:self.endX]]

cv2.dnn_registerLayer("Crop", CropLayer)

protoPath = "hed_model/deploy.prototxt"
modelPath = "hed_model/hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        img = cv2.imread(filepath)
        (H, W) = img.shape[:2]
        mean_pixel_values = np.average(img, axis=(0, 1))
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
                                     mean=(105, 117, 123),
                                     swapRB=False, crop=False)
        net.setInput(blob)
        hed = net.forward()
        hed = hed[0, 0, :, :]
        hed = (255 * hed).astype("uint8")

        # Connected component based labeling
        blur = cv2.GaussianBlur(hed, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)

        colors = np.random.randint(0, 255, size=(n_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        false_colors = colors[labels]

        MIN_AREA = 50
        detected = False
        for i, centroid in enumerate(centroids[1:], start=1):
            area = stats[i, 4]
            if area > MIN_AREA:
                detected = True
                cv2.drawMarker(false_colors, (int(centroid[0]), int(centroid[1])),
                               color=(255, 255, 255), markerType=cv2.MARKER_CROSS)

        original_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + filename)
        edge_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'edge_' + filename)
        segmented_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented_' + filename)
        final_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'final_' + filename)

        cv2.imwrite(original_img_path, img)
        cv2.imwrite(edge_img_path, hed)
        cv2.imwrite(segmented_img_path, false_colors)
        cv2.imwrite(final_img_path, false_colors)

        if detected:
            flash('Object(s) detected!')
        else:
            flash('All clear! No objects detected.')

        return redirect(url_for('upload_form', original='original_' + filename, edge='edge_' + filename, segmented='segmented_' + filename, final='final_' + filename))

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
