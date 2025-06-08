import os
import asyncio

from background import keep_alive
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder

from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, WebAppInfo
from aiogram import Bot
from aiogram.types import WebAppInfo, MenuButtonWebApp

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TOKEN or not GROQ_API_KEY:
    raise ValueError("Не заданы переменные окружения TOKEN или GROQ_API_KEY")

bot = Bot(token=TOKEN)
dp = Dispatcher()

MODEL_NAME = "d4data/biomedical-ner-all"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

web_app_url = "https://med-app-project-alpha.vercel.app/"  # ссылка на твоё веб-приложение

main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="Open Web", web_app=WebAppInfo(url=web_app_url))
        ]
    ],
    resize_keyboard=True
)

def prompt_chooser(lan: str) -> str:
    if lan == "en":
        return """
You are a professional medical assistant. Your task:

1) Analyze the symptoms found in the text.
2) Provide a detailed yet concise and serious response without jokes or fabrications.
3) Indicate only real existing specialists, from 1 to 3, relevant to the symptoms.
4) Never use inappropriate, offensive, or joking words.
5) Do not make a diagnosis.
6) If no symptoms are detected, do not indicate specialists, but give general health maintenance recommendations.
7) Consider that the user may describe both explicit symptoms and general complaints.
8) Format the response neatly and clearly with headings, numbered bullet points, and indentation.

---

Response format if symptoms are found:

*Detected symptoms:*
    1) [term 1] (type: [category])  
    2) [term 2] (type: [category])  
    ... (list all if there are many)

*Based on this data, I recommend consulting the following specialists:*
    1) [specialist 1]  
    2) [specialist 2]  
    3) [specialist 3]

*Recommendations:*
    1) [recommendation 1 — brief, clear, and medically neutral]  
    2) [recommendation 2]  
    3) [recommendation 3]
"""
    else:
        return """
Ты — профессиональный медицинский ассистент. Твоя задача:

1) Проанализировать симптомы, найденные в тексте.  
2) Выдать подробный, но лаконичный и серьёзный ответ без шуток и выдумок.  
3) Указывать только реально существующих специалистов (1–3), соответствующих симптомам.  
4) Никогда не использовать неуместные, оскорбительные или шуточные слова.  
5) Не ставить диагноз.  
6) Если симптомы не обнаружены — не указывай специалистов, а дай общие рекомендации по здоровью.  
7) Учитывай, что пользователь может описывать как симптомы, так и общие жалобы.  
8) Оформи ответ красиво и понятно — с подзаголовками, нумерацией и отступами.

---

Формат ответа, если симптомы найдены:

*Найденные симптомы:*  
    1) [термин 1] (тип: [категория])  
    2) [термин 2] (тип: [категория])  
    ... (если симптомов много — перечисли все)

*На основе этих данных рекомендую обратиться к специалистам:*  
    1) [специалист 1]  
    2) [специалист 2]  
    3) [специалист 3]

*Рекомендации:*  
    1) [рекомендация 1 — краткая, понятная и нейтральная]  
    2) [рекомендация 2]  
    3) [рекомендация 3]
"""
    return SYSTEM_PROMPT


language_message_ids = {}
user_languages = {}

class Form(StatesGroup):
    waiting_for_symptoms = State()

@dp.message(Command("start"))
async def start(message: types.Message):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="English \U0001F1FA\U0001F1F8", callback_data="lang_en"),
            InlineKeyboardButton(text="Русский \U0001F1F7\U0001F1FA", callback_data="lang_ru")
        ]
    ])
    sent_message = await message.answer(
        "👋 Please choose your language | Пожалуйста, выбери язык.",
        reply_markup=keyboard
    )
    await bot.set_chat_menu_button(
        menu_button=MenuButtonWebApp(
            text="Open",
            web_app=WebAppInfo(url=web_app_url)
        )
    )
    language_message_ids[message.from_user.id] = sent_message.message_id

@dp.callback_query(F.data.startswith("lang_"))
async def language_chosen(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    language = callback.data.split("_")[1]
    user_languages[user_id] = language

    if user_id in language_message_ids:
        try:
            await bot.delete_message(chat_id=user_id, message_id=language_message_ids[user_id])
        except Exception as e:
            print(f"Failed to delete message: {e}")

    await show_menu(callback)

async def show_menu(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_language = user_languages.get(user_id, "en")

    builder = InlineKeyboardBuilder()
    if user_language == 'en':
        builder.row(
            types.InlineKeyboardButton(text="\U0001F4F1 Open Web App", web_app=types.WebAppInfo(url=web_app_url))
        )
        builder.row(
            types.InlineKeyboardButton(text="\U0001F4AC Describe Symptoms", callback_data="describe_symptoms"),
            types.InlineKeyboardButton(text="\U0001F310 Change Language", callback_data="change_language")
        )
        text = """
\U0001F3E5 *Welcome to Med App!*

Here’s how I can help:
1. **Describe your symptoms** → Get AI-powered recommendations
2. **Book doctors** → Instant appointments via our Web App
3. **Change language**

Choose an option below or type your symptoms directly.
        """
    else:
        builder.row(
            types.InlineKeyboardButton(text="\U0001F4F1 Открыть Web App", web_app=types.WebAppInfo(url=web_app_url)),
        )
        builder.row(
            types.InlineKeyboardButton(text="\U0001F4AC Описать симптомы", callback_data="describe_symptoms"),
            types.InlineKeyboardButton(text="\U0001F310 Сменить язык", callback_data="change_language")
        )
        text = """
\U0001F3E5 *Добро пожаловать в Med App!*

Как я могу помочь:
1. **Опишите симптомы** → Получите рекомендации от ИИ
2. **Записаться к врачу** → Через наше Web-приложение
3. **Сменить язык**

Выберите действие ниже или напишите симптомы.
        """

    await callback.message.answer(text, parse_mode="Markdown", reply_markup=builder.as_markup())
    await callback.answer()

@dp.callback_query(F.data == "change_language")
async def change_language(callback: types.CallbackQuery):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="English \U0001F1FA\U0001F1F8", callback_data="lang_en"),
            InlineKeyboardButton(text="Русский \U0001F1F7\U0001F1FA", callback_data="lang_ru")
        ]
    ])
    sent_message = await callback.message.answer(
        "👋 Please choose your language | Пожалуйста, выбери язык.",
        reply_markup=keyboard
    )
    language_message_ids[callback.from_user.id] = sent_message.message_id
    await callback.answer()

@dp.callback_query(F.data == "describe_symptoms")
async def ask_for_symptoms(callback: types.CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    language = user_languages.get(user_id, "en")
    text = "\U0001F4DD Please describe your symptoms in detail." if language == 'en' else "\U0001F4DD Пожалуйста, опиши свои симптомы подробно."

    await state.set_state(Form.waiting_for_symptoms)
    await callback.message.answer(text)
    await callback.answer()

def ask_llm(prompt: str, user_language: str) -> str:
    SYSTEM_PROMPT = prompt_chooser(user_language)
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content

@dp.message(Form.waiting_for_symptoms)
async def analyze_symptoms(message: types.Message, state: FSMContext):
    await state.clear()  # Сброс состояния после получения текста

    user_id = message.from_user.id
    user_language = user_languages.get(user_id, "en")

    no_terms_text = "❗️ No medical terms detected." if user_language == 'en' else "❗️ Не удалось выявить медицинские термины."
    error_text = "🚫 Error analyzing symptoms." if user_language == 'en' else "🚫 Произошла ошибка при анализе."
    advice_text = "\n\n🤖 This bot gives advice but does not replace a doctor." if user_language == 'en' else "\n\n🤖 Бот даёт советы, но не заменяет врача."

    try:
        text = message.text
        entities = nlp(text)
        filtered = [ent for ent in entities if ent["entity_group"] != "O"]

        if not filtered:
            await message.answer(no_terms_text)
            return

        terms_text = "\n".join(f"- {ent['word']} (type: {ent['entity_group']})" for ent in filtered)
        prompt = f"{'The patient described' if user_language == 'en' else 'Пациент описал'}: {text}\n\n{'Recognized terms' if user_language == 'en' else 'Распознанные термины'}:\n{terms_text}"
        response = ask_llm(prompt, user_language)

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="\U0001F519 Back to Menu" if user_language == 'en' else "\U0001F519 Вернуться в меню", callback_data="lang_" + user_language)]
        ])

        await message.answer(response + advice_text, parse_mode="Markdown", reply_markup=keyboard)

    except Exception as e:
        await message.answer(error_text)
        print("Ошибка:", e)

keep_alive()
if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))