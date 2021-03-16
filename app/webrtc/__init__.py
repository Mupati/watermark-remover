from flask import Blueprint
webrtc = Blueprint('webrtc', '__init__')

from . import views  # isort:skip