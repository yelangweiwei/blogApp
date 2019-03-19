from django import forms
from comments.models import Comments

class CommentsForm(forms.ModelForm):
    class Meta:
        model = Comments   #指明表单对应的数据库模型是：Comments
        fields = ['name','email','url','text'] #要显示的字段
