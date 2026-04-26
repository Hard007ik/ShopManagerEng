import os
from pathlib import Path

from openenv.core.env_server import create_fastapi_app

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    def _load_dotenv(_path: str) -> bool:  # type: ignore[misc]
        return False


try:
    from .ShopManagerEng_environment import JewelryShopEnvironment
    from ..models import JewelryAction, JewelryObservation
except ImportError:
    from server.ShopManagerEng_environment import JewelryShopEnvironment
    from models import JewelryAction, JewelryObservation

import uvicorn

# Load .env from this package (ShopManagerEng/.env) for FRED/keys when running the server
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.is_file():
    _load_dotenv(_env)

# RL trainers (TRL GRPO, etc.) open one WebSocket per parallel rollout. With
# num_generations=8 + per_device_train_batch_size>=8 you can easily need 8-16
# concurrent envs. Default max is 1, so we bump it. Override via env var
# SHOPMANAGER_MAX_CONCURRENT_ENVS for hosted Spaces with tighter budgets.
_MAX_CONCURRENT_ENVS = int(os.environ.get("SHOPMANAGER_MAX_CONCURRENT_ENVS", "16"))

app = create_fastapi_app(
    JewelryShopEnvironment,
    JewelryAction,
    JewelryObservation,
    max_concurrent_envs=_MAX_CONCURRENT_ENVS,
)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()