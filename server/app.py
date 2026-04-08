from openenv.core.env_server import create_fastapi_app
from .ShopManagerEng_environment import JewelryShopEnvironment
from ..models import JewelryAction, JewelryObservation
import uvicorn

app = create_fastapi_app(JewelryShopEnvironment, JewelryAction, JewelryObservation)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)