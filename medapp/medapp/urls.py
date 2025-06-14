from django.urls import path
from medapp.users.views import telegram_auth_view

urlpatterns = [
    path("auth/telegram/", telegram_auth_view, name="telegram_auth"),
]