import hmac
import hashlib
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from django.contrib.auth import get_user_model
from django.contrib.auth import login as auth_login

User = get_user_model()
BOT_TOKEN = "7695389718:AAF7wojR4m8yK5J5Y89B_Umf_fwtvATVnw0"


def verify_telegram_webapp(data_check_string, received_hash):
    secret_key = hmac.new(
        "WebAppData".encode(), BOT_TOKEN.encode(), hashlib.sha256
    ).digest()
    computed_hash = hmac.new(
        secret_key, data_check_string.encode(), hashlib.sha256
    ).hexdigest()
    return computed_hash == received_hash


@csrf_exempt
def telegram_auth_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            init_data = data.get("hash")
            user_data = data

            # Проверяем данные Telegram WebApp
            if not verify_telegram_webapp(init_data, user_data.get("hash")):
                return JsonResponse({"error": "Invalid Telegram signature"}, status=403)

            # Создаем или получаем пользователя
            user, created = User.objects.get_or_create(
                username=f"tg_{user_data['id']}",
                defaults={
                    "first_name": user_data.get("first_name", ""),
                    "last_name": user_data.get("last_name", ""),
                },
            )

            # Авторизуем пользователя в Django
            auth_login(request, user)

            return JsonResponse(
                {
                    "status": "success",
                    "user": {
                        "id": user.id,
                        "username": user.username,
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                    },
                    "is_new_user": created,
                }
            )

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid method"}, status=405)
