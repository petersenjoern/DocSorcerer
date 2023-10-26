"""Main app"""

import api.router.question_answer as router_question_answer
import uvicorn
from fastapi import FastAPI

app = FastAPI(lifespan=router_question_answer.lifespan)
app.include_router(router_question_answer.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
