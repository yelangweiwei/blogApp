from django.contrib import admin
from userApp.models import Profile
# Register your models here.


class Profile_admin(admin.ModelAdmin):
    list_display = ['user','email']
admin.site.register(Profile,Profile_admin)
