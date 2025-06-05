import os
import asyncio

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from openai import OpenAI
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()
TOKEN = os.getenv("TOKEN")
if not TOKEN:
    raise ValueError("Не задан TOKEN")

# Telegram-бот
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Загрузка медицинской NER модели
MODEL_NAME = "d4data/biomedical-ner-all"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Подключение к локальной LLM через Ollama (например, Mistral)
client = OpenAI(
    base_url="http://localhost:11434/v1",  # адрес Ollama
    api_key="ollama"  # фиктивный ключ
)

# Системный промпт
SYSTEM_PROMPT = """
Ты — профессиональный медицинский ассистент. Твоя задача:

1. Проанализировать симптомы, найденные в тексте.
2. Выдать подробный, но лаконичный и серьёзный ответ без шуток и выдумок.
3. Указывать только реально существующих специалистов, от 1 до 3, соответствующих симптомам.
4. Никогда не использовать неуместные, оскорбительные или шуточные слова.
5. Не ставить диагноз.
6. Если симптомы не обнаружены, не указывай специалистов, а дай общие рекомендации для сохранения здоровья.
7. Учитывай, что пользователь может описать как явные симптомы, так и общие жалобы.
8. Форматируй ответ аккуратно и читабельно с подзаголовками и пунктами.

---

Формат ответа Если найдены симптомы:

Найденные симптомы:
    - [термин 1] (тип: [категория])
    - [термин 2] (тип: [категория])
    ... (если симптомов много — перечисли все)

На основе этих данных рекомендую обратиться к следующим специалистам:
    - [специалист 1]
    - [специалист 2] (если есть)
    - [специалист 3] (если есть)

Рекомендации:
    1. [рекомендация 1 — краткая, понятная и медицински нейтральная]
    2. [рекомендация 2] (если есть)
    3. [рекомендация 3] (не обязательно все три, можно меньше)
"""


@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer(
        "👋 Привет! Я помогу определить возможные симптомы в твоем описании. "
        "Просто расскажи, что тебя беспокоит."
    )


# Отправка запроса к локальной модели
def ask_llm(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="mistral",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content


@dp.message()
async def analyze_symptoms(message: types.Message):
    try:
        text = message.text
        entities = nlp(text)
        filtered = [ent for ent in entities if ent["entity_group"] != "O"]

        if not filtered:
            await message.answer(
                "❗️ Не удалось выявить медицинские термины. Попробуйте описать симптомы подробнее."
            )
            return

        # Формируем строку из распознанных сущностей
        terms_text = "\n".join(
            f"- {ent['word']}" for ent in filtered
        )

        prompt = f"Пациент описал: {text}\n\nРаспознанные термины:\n{terms_text}"
        response = ask_llm(prompt)

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔗 Официальный медицинский ресурс", url="https://vk.com/video807566_169118280")]
        ])

        await message.answer(response +
            "\n\nНаш бот 🤖 даёт советы 📝, но не заменяет врача 🩺. При любых сомнениях или ухудшениях состояния обязательно проконсультируйся со специалистом.",
            parse_mode="Markdown",
            reply_markup=keyboard
        )

    except Exception as e:
        await message.answer("🚫 Произошла ошибка при анализе. Попробуйте позже.")
        print("Ошибка:", e)


if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))