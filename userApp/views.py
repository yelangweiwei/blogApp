from django.shortcuts import render,redirect
from userApp.models import Profile
from userApp.forms import RegisterForm
# Create your views here.

def register(request):
    #只有post时，才表示用户提交了注册了信息
    if request.method=='POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            #如果数据提交的合法
            form.save()
            return redirect('/')
    else:
        #当请求不是post时，标明用户正在访问注册页面，返回一个空的表单给用户
        form = RegisterForm()
        return render(request,'userApp/register.html',context={'form':form})
