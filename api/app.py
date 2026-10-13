"""FastAPI app and the Lambda entry point (`api.app.handler`)."""

from fastapi import FastAPI
from mangum import Mangum

from api.errors import install_error_handlers
from api.routes import me

app = FastAPI(title="leaf-al API", docs_url=None, redoc_url=None, openapi_url=None)
install_error_handlers(app)
app.include_router(me.router)

handler = Mangum(app, lifespan="off")
