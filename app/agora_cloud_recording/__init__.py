from flask import Blueprint
agora_cloud_recording = Blueprint('agora_cloud_recording', '__init__')

from . import views  # isort:skip