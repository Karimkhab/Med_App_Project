import hmac
import hashlib
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from django.contrib.auth import get_user_model

User = get_user_model()
BOT_TOKEN = "7695389718:AAF7wojR4m8yK5J5Y89B_Umf_fwtvATVnw0"

def verify_telegram_auth(data: dict, received_hash: str) -> bool:
    auth_data = [f"{k}={v}" for k, v in sorted(data.items()) if k != "hash"]
    data_check_string = "\n".join(auth_data)
    secret_key = hashlib.sha256(BOT_TOKEN.encode()).digest()
    hmac_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    return hmac_hash == received_hash

@csrf_exempt
def telegram_auth_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        hash_from_client = data.get("hash")

        auth_data = {
            "id": data.get("id"),
            "first_name": data.get("first_name"),
            "last_name": data.get("last_name"),
            "username": data.get("username"),
        }

        if not verify_telegram_auth(auth_data, hash_from_client):
            return JsonResponse({"error": "Invalid Telegram signature"}, status=403)

        # Автоматическая регистрация/авторизация
        user, created = User.objects.get_or_create(username=str(auth_data["id"]), defaults={
            "first_name": auth_data.get("first_name", ""),
            "last_name": auth_data.get("last_name", ""),
        })

        return JsonResponse({
            "status": "ok",
            "user_id": user.id,
            "created": created
        })

    return JsonResponse({"error": "Invalid method"}, status=405)