from __future__ import annotations

from pathlib import Path

from flask import Flask, send_from_directory

from service.flask.routes.customer import customer_bp
from service.flask.routes.predict import predict_bp
from service.flask.routes.stats import stats_bp

# 项目根目录（service/flask/app.py 向上两级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DASHBOARD_DIR = str(_PROJECT_ROOT / "dashboard")


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(customer_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(stats_bp)

    @app.get("/")
    def dashboard():
        return send_from_directory(_DASHBOARD_DIR, "index.html")

    @app.get("/dashboard/<path:filename>")
    def dashboard_assets(filename: str):
        return send_from_directory(_DASHBOARD_DIR, filename)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
