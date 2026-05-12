from src.config import get_config
from src.realtime_api import create_app


def main() -> None:
    cfg = get_config()
    app = create_app(cfg)
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()

