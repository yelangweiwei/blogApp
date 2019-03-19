from django.conf.urls import url
from Blog.views import index,detail,archives,categories,postListOfcategories,\
    allPostByCreateTime,allPostByAuthorId,IndexView,CategoryView,ArchivesView,\
    PostDetailView,TagView,search

app_name = 'Blog'
urlpatterns = [
    url(r'^welcome/',IndexView.as_view(),name='index'),
    # url(r'^welcome/',index,name='index'),
    # url(r'^post/(?P<pk>[0-9]+)/$',detail,name='detail'),  #命名捕获组
    url(r'^post/(?P<pk>[0-9]+)/$',PostDetailView.as_view(),name='detail'),  #命名捕获组
    url(r'^archives/(?P<year>[0-9]{4})/(?P<month>[0-9]{1,2})/$',ArchivesView.as_view(),name='archives'),
    # url(r'^archives/(?P<year>[0-9]{4})/(?P<month>[0-9]{1,2})/$',archives,name='archives'),
    url(r'^categories/(?P<pk>[0-9]+)/$',CategoryView.as_view(),name='categories'),
    # url(r'^categories/(?P<pk>[0-9]+)/$',categories,name='categories'),
    url(r'^allcategoriesList/(?P<pk>[0-9]+)/$',postListOfcategories,name='postListOfcategories'),
    url(r'^allPostByCreateTime/(?P<year>[0-9]{4})/(?P<month>[0-9]{1,2})/(?P<day>[0-9]{1,2})/$',allPostByCreateTime,name = 'allPostByCreateTime'),
    url(r'^allPostByAuthorId/(?P<pk>[0-9]+)/$',allPostByAuthorId,name='allPostByAuthorId'),
    url(r'^tags/(?P<pk>[0-9]+)/$',TagView.as_view(),name='tags'),
    # url(r'^search/$',search,name='search'),

]