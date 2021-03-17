import os
from flask import Flask, g
from flask_login import LoginManager, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate, current

from .errors import Errors

# init SQLAlchemy so we can use it later in our models
db = SQLAlchemy()
login_manager = LoginManager()


def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
        'SQLALCHEMY_DATABASE_URI')

    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = os.environ.get(
        'SQLALCHEMY_TRACK_MODIFICATIONS')

    login_manager.init_app(app)
    login_manager.login_message = "Access Denied!, You are not Authenticated"
    login_manager.login_view = "auth.login"

    db.init_app(app)
    migrate = Migrate(app, db, compare_type=True)

    # blueprint for auth routes in our app
    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    # blueprint for home
    from .home import home as home_blueprint
    app.register_blueprint(home_blueprint)

    # blueprint for agora
    from .agora import agora as agora_blueprint
    app.register_blueprint(agora_blueprint)

    with app.app_context():
        # blueprint for watermark remover
        from .watermark_remover import watermark_remover as watermark_remover_blueprint
        app.register_blueprint(watermark_remover_blueprint)

    # blueprint from webrtc apps
    from .webrtc import webrtc as webrtc_blueprint
    app.register_blueprint(webrtc_blueprint)

    app.register_error_handler(404, Errors.page_not_found)
    app.register_error_handler(403, Errors.forbidden)
    app.register_error_handler(500, Errors.internal_server_error)

    @app.before_request
    def before_request():
        if current_user.is_authenticated:
            g.user = current_user

    return app
