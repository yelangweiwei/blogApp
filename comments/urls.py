from django.conf.urls import url
from comments.views import post_comment

app_name = 'comments'
urlpatterns=[
    url(r'^comment/post/(?P<post_pk>[0-9]+)/$',post_comment,name='post_comment'),
]