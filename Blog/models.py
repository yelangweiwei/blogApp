import markdown
from django.db import models
from django.contrib.auth.models import User
# from userApp.models import User
from django.utils.six import python_2_unicode_compatible
from django.urls import reverse
from django.utils.html import strip_tags


# Create your models here.


#分类
@python_2_unicode_compatible
class Category(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name
@python_2_unicode_compatible
class Tag(models.Model):
    name = models.CharField(max_length=100)
    def __str__(self):
        return self.name
@python_2_unicode_compatible
class Post(models.Model):
    #文章标题
    title = models.CharField(max_length=70)
    #文章的正文
    body = models.TextField()
    #文章的创建事件和最近的更改时间
    created_time = models.DateTimeField()
    modified_time = models.DateTimeField()
    #文章摘要，可以没有文章摘要，在默认情况下，charfield可以为空
    excerpt = models.CharField(max_length=200,blank=True)

    #分类和标签，数据库表关联起来
    category = models.ForeignKey(Category)
    tags = models.ManyToManyField(Tag,blank=True)

    #文章的作者是经过认证的用户
    author = models.ForeignKey(User)
    #用于记录阅读量
    views = models.PositiveIntegerField(default=0)#只允许为整数或者为0
    def increase_view(self):
        self.views+=1
        self.save(update_fields=['views'])  #告诉django只更新数据库中的views字段的值；不是精确的统计阅读量，偶尔冲突的不计

    def __str__(self):
        return ('%s %s %s %s %s %s')%(self.title,self.category,self.tags,self.author,self.created_time,self.modified_time)

    def get_absolute_url(self):
        return reverse('Blog:detail',kwargs={'pk':self.pk})

    class Meta:
        ordering=['-created_time']

    def save(self,*args,**kwargs):
        #没有写摘要
        if not self.excerpt:
            #先渲染一个markdown类，用于渲染body的文章
            md = markdown.markdown(extensions=[
                'markdown.extensions.extra',
                'markdown.extensions.codehilite',
            ])
            #先将markdown文章渲染成html文本
            #使用strip_tags 去掉html文本的标签
            #从文本中抽取54个字节给excerpt
            self.excerpt = strip_tags(md.convert(self.body))[:54]
        super(Post,self).save(*args,**kwargs)  #将数据保存到数据库










