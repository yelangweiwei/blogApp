from django.shortcuts import render,redirect
from userApp.forms import RegisterForm
# Create your views here.

def register(request):
    #只有post时，才表示用户提交了注册了信息
    #从get或者post中获取next的参数值
    #get中；next通过/?next=value
    #post 通过表单传递
    redirect_to = request.POST.get('next',request.GET.get('next',''))

    if request.method=='POST':
        #request.POST是一个类字典数据结构，记录了了用户提交的注册信息
        #实例化
        form = RegisterForm(request.POST)
        if form.is_valid():
            #如果数据提交的合法
            form.save()
            if redirect_to:
                return redirect(redirect_to)
            else:
                return redirect('/')
    else:
        #当请求不是post时，标明用户正在访问注册页面，返回一个空的表单给用户
        form = RegisterForm()
    #如果用户正在访问注册页面， 则则渲染一个空的注册表单
    #如果用户通过通过表单传递信息，但是数据验证不合法，则渲染一个常有错误信息的表单
    #将记录用户注册前页面的redirect_to传递给模板，以维持next参数在整个注册过程中的传递
    return render(request,'userApp/register.html',context={'form':form,'next':redirect_to})


def loginIndex(request):
    return render(request,'userApp/loginIndex.html')