from flask import render_template

from . import webrtc


@webrtc.route('/webrtc')
def index():
    return render_template('webrtc/index.html', title="Webrtc")
