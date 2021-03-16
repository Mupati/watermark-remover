from flask import request, render_template, jsonify

from PIL import Image, ImageEnhance

from flask.helpers import send_from_directory, url_for
from pdf2image import convert_from_path
import tempfile
import os
from werkzeug.utils import secure_filename

from . import watermark_remover
from .. import create_app

flask_app = create_app()


UPLOAD_FOLDER = '../uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}


flask_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 20MB file size limit
flask_app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# Image Processing Functions


def pdf_to_images(pdf_path: str) -> list:
    with tempfile.TemporaryDirectory() as path:
        pil_images = convert_from_path(pdf_path, output_folder=path)
        return pil_images


def increase_contrast(pil_image):
    img = ImageEnhance.Contrast(pil_image).enhance(2.8)
    pil_image = img.convert('RGBA')
    return pil_image


def remove_watermark_with_pil(pil_images: list) -> None:
    res = []
    for img in pil_images:
        im = increase_contrast(img)
        R, G, B = im.convert('RGB').split()
        r = R.load()
        g = G.load()
        b = B.load()
        w, h = im.size

        # Convert non-black pixels to white
        for i in range(w):
            for j in range(h):
                if(r[i, j] > 100 or g[i, j] > 100 or b[i, j] > 100):
                    r[i, j] = 255  # Just change R channel

        # Merge just the R channel as all channels
        im = Image.merge('RGB', (R, R, R))
        res.append(im)
    return res


def convert_images_to_pdf(pil_images: list, filename):
    processed_file_path = os.path.join(
        flask_app.config['UPLOAD_FOLDER'], 'processed_'+filename+'.pdf')
    pil_images[0].save(processed_file_path, save_all=True,
                       append_images=pil_images[1:])
    return 'processed_'+filename+'.pdf'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Routes and Endpoints
@watermark_remover.route('/watermark')
def index():
    return render_template('watermark_remover/index.html')


@watermark_remover.route('/watermark/upload/<path:filename>')
def processed_file_dir(filename):
    return send_from_directory(flask_app.config['UPLOAD_FOLDER'], filename)


@watermark_remover.route('/watermark/process', methods=['POST'])
def proccess_file():

    # Create uploads folder
    if not os.path.exists('../uploads'):
        os.mkdir('../uploads')

    # check if the post request has the file part
    if 'file' not in request.files:
        return jsonify(status="error", message="Missing image or PDF file"), 422

    uploadedFile = request.files['file']

    if uploadedFile.filename == '':
        return jsonify(status="error", message="No selected file"), 422

    if uploadedFile and allowed_file(uploadedFile.filename):
        filename = secure_filename(uploadedFile.filename)
        uploadedFile.save(os.path.join(
            flask_app.config['UPLOAD_FOLDER'], filename))

        stored_file = os.path.join(flask_app.config['UPLOAD_FOLDER'], filename)

        pil_images = pdf_to_images(pdf_path=stored_file)
        images_no_watermarks = remove_watermark_with_pil(pil_images=pil_images)

        name_without_extension = filename.split('.')[0]
        processed_file = convert_images_to_pdf(
            images_no_watermarks, name_without_extension)

        # After Processing, delete the uploaded file
        if os.path.exists(stored_file):
            os.remove(stored_file)

        # Return the path to the processed file to be downloaded
        return jsonify(status="success", message="Conversion complete",
                       processed_file_path=url_for('watermark_remover.processed_file_dir', filename=processed_file)), 200
    else:
        return jsonify(status="error", message="Supported files are: pdf, jpg, png or jpeg"), 422
