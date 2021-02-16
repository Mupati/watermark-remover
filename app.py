from flask import Flask, request, render_template, jsonify

from PIL import Image, ImageEnhance
from pdf2image import convert_from_path
import tempfile


app = Flask(__name__)


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
        # im = Image.open(filename)
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


def convert_images_to_pdf(pil_images: list):
    pil_images[0].save('cleaned_score.pdf', save_all=True,
                       append_images=pil_images[1:])

# Routes and Endpoints


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def proccess_file():
    image_file = request.files['file']
    image_file.save("./"+image_file.filename)
    return jsonify(status="Ok", message="Conversion complete")


if __name__ == "__main__":
    app.run(debug=True)
