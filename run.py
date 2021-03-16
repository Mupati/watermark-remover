import os

from app import create_app
config_name = os.environ.get('FLASK_ENV')

flask_app = create_app()

if __name__ == '__main__':
    flask_app.run(debug=os.environ.get('ENABLE_DEBUG'))
