from django.conf.urls import url
from userApp.views import register,loginIndex

app_name = 'userApp'
urlpatterns = [
    url(r'^register/$',register,name='register'),
    url(r'^loginIndex/$',loginIndex,name='loginIndex')
]