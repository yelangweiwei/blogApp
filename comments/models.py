from django.db import models
from django.utils.six import  python_2_unicode_compatible
# Create your models here.

@python_2_unicode_compatible
class Comments(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(max_length=100)
    url = models.URLField(blank=True)
    text = models.TextField()
    created_time = models.DateTimeField(auto_now_add=True)
    post = models.ForeignKey('Blog.Post')
    def __str__(self):
        return self.text[:20]
