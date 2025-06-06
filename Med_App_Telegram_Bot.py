import os
import asyncio

from background import keep_alive
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from openai import OpenAI  # ✅ Groq тоже использует OpenAI-клиент
from dotenv import load_dotenv

from aiogram import F

load_dotenv()
TOKEN = os.getenv("TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # ✅ Новый ключ

if not TOKEN or not GROQ_API_KEY:
    raise ValueError("Не заданы переменные окружения TOKEN или GROQ_API_KEY")

bot = Bot(token=TOKEN)
dp = Dispatcher()

# Модель для NER
MODEL_NAME = "d4data/biomedical-ner-all"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"  # ✅ Groq endpoint
)

user_languages = {}

def prompt_chooser(lan: str) -> str:
    if lan == "en":
        SYSTEM_PROMPT = """
You are a professional medical assistant. Your task:

1. Analyze the symptoms found in the text.
2. Provide a detailed yet concise and serious response without jokes or fabrications.
3. Indicate only real existing specialists, from 1 to 3, relevant to the symptoms.
4. Never use inappropriate, offensive, or joking words.
5. Do not make a diagnosis.
6. If no symptoms are detected, do not indicate specialists, but give general health maintenance recommendations.
7. Consider that the user may describe both explicit symptoms and general complaints.
8. Format the response neatly and clearly with headings and bullet points.

---

Response format if symptoms are found:

Detected symptoms:
    - [term 1] (type: [category])
    - [term 2] (type: [category])
    ... (if there are many symptoms — list them all)

Based on this data, I recommend consulting the following specialists:
    - [specialist 1]
    - [specialist 2]
    - [specialist 3]

Recommendations:
    1. [recommendation 1 — brief, clear, and medically neutral]
    2. [recommendation 2] (if any)
    3. [recommendation 3] (not necessarily all three, can be fewer)
"""
    else:
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
    - [специалист 2]
    - [специалист 3]

Рекомендации:
    1. [рекомендация 1 — краткая, понятная и медицински нейтральная]
    2. [рекомендация 2] (если есть)
    3. [рекомендация 3] (не обязательно все три, можно меньше)
"""
    return SYSTEM_PROMPT



@dp.callback_query(F.data.startswith("lang_"))
async def language_chosen(callback: types.CallbackQuery):
    lang = callback.data.split("_")[1]  # 'en' или 'ru'
    user_languages[callback.from_user.id] = lang

    greetings = {
        'en': "Great! Now you can send me your symptoms description in English.",
        'ru': "Отлично! Теперь ты можешь описывать симптомы на русском."
    }
    await callback.message.answer(greetings[lang])
    await callback.answer()  # Убирает "часики" у кнопки


@dp.message(Command("start"))
async def start(message: types.Message):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="English 🇺🇸", callback_data="lang_en"),
            InlineKeyboardButton(text="Русский 🇷🇺", callback_data="lang_ru")
        ]
    ])
    await message.answer(
        "👋 Hello! I will help you identify possible symptoms in your description. But first, please choose the language you want to communicate in.\n\n👋 Привет! Я помогу определить возможные симптомы в твоём описании. Но сначала выбери, пожалуйста, язык, на котором ты хочешь общаться.",reply_markup=keyboard
    )

# @dp.message(Command("start"))
# async def start(message: types.Message):
#     await message.answer(
#         "👋 Привет! Я помогу определить возможные симптомы в твоем описании. "
#         "Просто расскажи, что тебя беспокоит."
#     )


def ask_llm(prompt: str, lang) -> str:
    SYSTEM_PROMPT = prompt_chooser(lang)
    completion = client.chat.completions.create(
        model="llama3-70b-8192",  # ✅ Модель от Groq
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content


@dp.message()
async def analyze_symptoms(message: types.Message):
    lang = user_languages.get(message.from_user.id, 'ru')

    if lang == 'en':
        no_terms_text = "❗️ No medical terms were detected. Please describe your symptoms in more detail."
        error_text = "🚫 An error occurred during analysis. Please try again later."
        advice_text = "\n\nOur bot 🤖 provides advice 📝 but does not replace a doctor 🩺. In case of doubts or worsening, consult a specialist."
    else:
        no_terms_text = "❗️ Не удалось выявить медицинские термины. Попробуйте описать симптомы подробнее."
        error_text = "🚫 Произошла ошибка при анализе. Попробуйте позже."
        advice_text = "\n\nНаш бот 🤖 даёт советы 📝, но не заменяет врача 🩺. При любых сомнениях или ухудшениях состояния обязательно проконсультируйся со специалистом."

    try:
        text = message.text
        entities = nlp(text)
        filtered = [ent for ent in entities if ent["entity_group"] != "O"]

        if not filtered:
            await message.answer(no_terms_text)
            return

        terms_text = "\n".join(f"- {ent['word']}" for ent in filtered)
        prompt = f"Пациент описал: {text}\n\nРаспознанные термины:\n{terms_text}"

        response = ask_llm(prompt, lang)

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text="🔗 Official medical resource" if lang == 'en' else "🔗 Официальный медицинский ресурс",
                url="https://vk.com/video807566_169118280")]
        ])

        await message.answer(response + advice_text, parse_mode="Markdown", reply_markup=keyboard)

    except Exception as e:
        await message.answer(error_text)
        print("Ошибка:", e)

keep_alive()
if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))