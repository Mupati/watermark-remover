from flask import render_template
import app


@app.route('/')
@app.route('/index')
def index():
    return render_template('home/index.html', title='HOME')
