from flask import Blueprint

watermark_remover = Blueprint('watermark_remover', '__init__')

from . import views  # isort:skip