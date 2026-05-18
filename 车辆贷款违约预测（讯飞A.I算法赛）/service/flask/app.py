from __future__ import annotations

import threading
import time
from pathlib import Path

from flask import Flask, send_from_directory

from service.flask.repositories import similarity_engine
from service.flask.routes.customer import customer_bp
from service.flask.routes.import_pipeline import import_bp
from service.flask.routes.model_explain import model_explain_bp
from service.flask.routes.predict import predict_bp
from service.flask.routes.repair import repair_bp
from service.flask.routes.stats import stats_bp

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DASHBOARD_DIR = str(_PROJECT_ROOT / "dashboard")

_metrics_lock = threading.Lock()
METRICS = {"total_requests": 0, "total_latency_ms": 0.0}
_tls = threading.local()


def get_metrics() -> dict:
    with _metrics_lock:
        return dict(METRICS)


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(customer_bp)
    app.register_blueprint(import_bp)
    app.register_blueprint(model_explain_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(repair_bp)
    app.register_blueprint(stats_bp)

    # 后台线程预加载相似客户检索引擎（不阻塞 Flask 启动）
    threading.Thread(
        target=similarity_engine.load, daemon=True,
        name="similarity-loader",
    ).start()

    @app.before_request
    def _before():
        _tls._req_start = time.time()

    @app.after_request
    def _after(response):
        start = getattr(_tls, "_req_start", None)
        if start is not None:
            elapsed = (time.time() - start) * 1000
            del _tls._req_start
            with _metrics_lock:
                METRICS["total_requests"] += 1
                METRICS["total_latency_ms"] += elapsed
        return response

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
