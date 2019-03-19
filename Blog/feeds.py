from django.contrib.syndication.views import Feed
from Blog.models import Post

class AllPostsRssFeed(Feed):
    #显示在聚合器上的标题
    title = 'Django 博客教程演示教程'
    #通过聚合阅读器跳转到网站的地址
    link = '/'
    #显示在聚合阅读器上的描述信息
    description = 'Django 博客教程演示项目测试文章'

    #需要显示的内容条目
    def items(self):
        return Post.objects.all()
    #聚合器中显示的内容的条目的标题
    def item_title(self,item):
        return '[%s]%s' %(item.category,item.title)

    #聚合器中显示的内容条目的描述  没用，在聚合器中，应该是给屏蔽了，在聚合器中直接指定到网址指定的位置
    # def item_description(self, item):
    #     return item.body[:100]