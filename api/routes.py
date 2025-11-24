from fastapi import APIRouter
from api.schemas import AskRequest, AskResponse
from rag.pipeline import generate_answer

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answer = generate_answer(req.question)
    return AskResponse(answer=answer)