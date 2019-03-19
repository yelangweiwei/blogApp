from django.contrib.auth.forms import UserCreationForm
from userApp.models import Profile
from django.contrib.auth.models import User

class RegisterForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username","email")


