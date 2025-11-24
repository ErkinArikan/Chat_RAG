from config import settings
from services.openai_client import client
from rag.retriever import get_relevant_chunks


def generate_answer(question: str) -> str:
    print("\n\n[DEBUG] Gelen soru:", question)

    chunks = get_relevant_chunks(question)
    context = "\n\n---\n\n".join(chunks)

    print("[DEBUG] Seçilen context parçaları:\n", context, "\n")

    system_prompt = (
        "Her yeni sohbet başladığında albinasoft insan kaynakları departmanı yöneticisiyim "
        "sizlere nasıl yardımcı olabilirim diye sorar. "
        "Sen şirket içi bir doküman asistanısın. "
        "Sadece aşağıdaki doküman parçalarına dayanarak cevap ver. "
        "Önce doküman parçalarını dikkatlice oku. "
        "Soruya kısmen bile cevap verebilen bilgiler varsa, bunları kullanarak ve alıntı yaparak cevap ver. "
        "Eğer dokümanda sadece kısmi bilgi varsa, 'Bu dokümana göre şu kadarını biliyorum: ...' gibi açıkla. "
        "Gerçekten hiçbir ilgili bilgi yoksa 'Bu dokümanda bu bilgi yok.' de. "
        "Uydurma veya tahmin yürütme, sadece verilen metni kullan."
    )

    user_content = (
        f"Soru:\n{question}\n\n"
        f"İlgili doküman parçaları:\n{context}"
    )

    print("[DEBUG] SYSTEM PROMPT GPT'ye giden:\n", system_prompt, "\n")
    print("[DEBUG] USER CONTENT GPT'ye giden (soru + context):\n", user_content, "\n")

    completion = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    answer = completion.choices[0].message.content

    try:
        print("[DEBUG][CHAT USAGE]:", completion.usage)
    except Exception:
        pass

    print("[DEBUG] GPT cevabı:\n", answer, "\n")

    return answer