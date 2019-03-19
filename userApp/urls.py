from django.conf.urls import url
from userApp.views import register

app_name = 'userApp'
urlpattern = [
    url(r'^register/$',register,name='register'),
]