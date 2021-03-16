from flask import render_template


class Errors():
    def forbidden(error):
        return render_template('error/403.html', title='Forbidden'), 403

    def page_not_found(error):
        return render_template('error/404.html', title='Page Not Found'), 404

    def internal_server_error(error):
        return render_template('error/500.html', title='Server Error'), 500
