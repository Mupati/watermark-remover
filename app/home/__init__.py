from flask import Blueprint

home = Blueprint('home', '__init__')

from . import views  # isort:skip