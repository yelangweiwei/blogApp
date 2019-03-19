import markdown
from django.shortcuts import render,get_object_or_404,get_list_or_404
from Blog.models import Post,Category,Tag
from comments.forms import CommentsForm
from django.views.generic import ListView,DetailView
from django.core.paginator import Paginator,EmptyPage,PageNotAnInteger
from django.utils.text import slugify
from markdown.extensions.toc import TocExtension
from django.db.models import Q


# Create your views here.
class IndexView(ListView):
    model = Post  #指明model
    template_name = 'blog/index.html' #要渲染的模板
    context_object_name = 'post_list' #指定获取的模型列表的数据保存的变量名，这个变量的会传给模板
    #指定paginate_by 属性后的分页功能，其值代表每一页包含多少篇文章
    paginate_by = 1

    def get_context_data(self, **kwargs):
        #首先获得父类生成的传递给模板的字典
        context = super().get_context_data(**kwargs)
        paginator = context.get('paginator')
        page = context.get('page_obj')
        is_paginated = context.get('is_paginated')

        #调用自己写的pagination_data 方法显示分页导航条需要的数据
        pagination_data = self.pagination_data(paginator,page,is_paginated)

        #将分页导航条的模板变量更新到context中，
        context.update(pagination_data)
        return context
    def pagination_data(self,paginator,page,is_paginated):
        if not is_paginated:  #没有分页，就无法显示分页导航
            return {}
        #当前页左边的页码号，初始值为空
        left = []
        #当前页右边的页码号，初始值为空
        right = []

        #标识第一页的后边是否需要省略号
        right_has_more = False
        #标识最后一页的前边是否需要省略号
        left_has_more = False

        #标识第一页是否显示
        first= False
        #标识最后一页是否显示
        last = False
        #获得当前页的页码
        page_number = page.number
        #获得总的页码
        total_pages = paginator.num_pages
        #获得整个分页页码列表，比如分了四页，[1,2,3,4]
        page_range = paginator.page_range
        #如果当前页是第一页
        if page_number==1:
            right = page_range[page_number:page_number+2]
            #如果右边的最后一页比总页数减去1还要小，说明最右边还有页码号
            if right[-1]<total_pages-1:
                right_has_more = True
            if right[-1]<total_pages: #最后一页显示
                last = True
        elif page_number == total_pages:
            left = page_range[page_number-3 if page_number-3>0 else 0:page_number-1]
            if left[0]> 2:
                left_has_more=True
            if left[0] >1:
                first = True  #显示第一页
        else:
            left = page_range[page_number - 3 if page_number - 3 > 0 else 0:page_number - 1]
            if left[0] > 2:
                left_has_more = True
            if left[0] > 1:
                first = True

            right = page_range[page_number:page_number + 2]
            if right[-1] < total_pages - 1:
                right_has_more = True
            if right[-1] < total_pages:  # 最后一页显示
                last = True
        data = {
            'left':left,
            'right':right,
            'left_has_more':left_has_more,
            'right_has_more':right_has_more,
            'first':first,
            'last':last
        }
        return data



def index(request):
    post_list = Post.objects.all()
    paginator = Paginator(post_list,1)
    page = request.GET.get('page')
    try:
        postList = paginator.page(page)
    except PageNotAnInteger:
        postList = paginator.page(1)
    except EmptyPage:
        postList = paginator.page(1)
    return render(request,'blog/index.html',context={
        'title':'我的博客首页',
        'welcome':'欢迎访问我的博客首页',
        'post_list':postList}
                  )

#
# def index(request):
#     print('-------------000')
#     post_list = Post.objects.all()
#     return render(request, 'blog/index.html', context={
#         'title':'我博客首页',
#         'welcome':'欢迎访问我的博客首页',
#         'post_list':post_list
#     })


class PostDetailView(DetailView):
    model = Post
    template_name = 'blog/detail.html'
    context_object_name = 'post'

    def get(self,request,*args,**kwargs):
        #复写get方法，get方法返回一个httpReponse实例，只有在调用了get方法后，才有self.object 属性，其值是Post模型实例，即被访问的文章post
        response =  super().get(request,*args,**kwargs)
        #阅读量增加increase_view
        self.object.increase_view()
        return response
    def get_object(self, queryset=None):
        #获得post的对象，并对body进行渲染
        post = super().get_object(queryset=None)
        # post.body = markdown.markdown(post.body,
        #                               extensions={
        #                                   'markdown.extensions.extra',
        #                                   'markdown.extensions.codehilite',
        #                                   'markdown.extensions.toc',
        #                               })
        md = markdown.Markdown(extensions=['markdown.extensions.extra',
                                           'markdown.extensions.codehilite',
                                           # 'markdown.extensions.toc',
                                           TocExtension(slugify=slugify)  #是个实例，slugify作为函数，用于处理标题的锚点值，slugify处理中文
                                           ])
        post.body = md.convert(post.body)
        post.toc = md.toc
        return post

    def get_context_data(self, **kwargs):
        context  = super().get_context_data(**kwargs)
        form = CommentsForm()
        comment_list = self.object.comments_set.all()
        context.update(
            {
                'form':form,
                'comment_list':comment_list
            }
        )
        return context



def detail(request,pk):
    post= get_object_or_404(Post,pk=pk)#这里是个字典

    #阅读量+1
    post.increase_view()

    #使用markdown进行渲染body
    post.body = markdown.markdown(post.body,
                                  extensions=[
                                      'markdown.extensions.extra',#缩写，表格的扩展
                                      'markdown.extensions.codehilite',#高亮
                                      'markdown.extensions.toc',#边侧边栏目录，在文档中产生目录
                                  ])
    #导入commentform
    form = CommentsForm()
    #获得post下的全部评论
    comments_list = post.comments_set.all()
    comments_count = post.comments_set.count()
    return render(request,'blog/detail.html',context={
        'post':post,
        'form':form,
        'comment_list':comments_list,
        'comment_count':comments_count
    })


class ArchivesView(ListView):
    model = Post
    template_name = 'blog/index.html'
    context_object_name = 'post_list'
    def get_queryset(self):
        return super(ArchivesView,self).get_queryset().filter(created_time__year=self.kwargs.get('year'),created_time__month=self.kwargs.get('month'))

def archives(request,year,month):
    post_list = Post.objects.filter(created_time__year=year,created_time__month=month)
    return render(request,'blog/index.html',context={'post_list':post_list})


class CategoryView(ListView):
    model = Post
    template_name = 'blog/index.html'
    context_body_name= 'post_list'
    def get_queryset(self):  #默认是获取模型的全部列表数据，重写该方法
        cate = get_object_or_404(Category,pk=self.kwargs.get('pk'))
        return super(CategoryView,self).get_queryset().filter(category=cate)

class TagView(ListView):
    model = Post
    template_name = 'blog/index.html'
    context_object_name = 'post_list'
    def get_queryset(self):
        tag = get_object_or_404(Tag,pk=self.kwargs.get('pk'))
        return super(TagView,self).get_queryset().filter(tags=tag)


def categories(request,pk):
    cate = get_object_or_404(Category,pk=pk)
    post_list = Post.objects.filter(category=cate)
    return render(request,'blog/index.html',context={'post_list':post_list})

def postListOfcategories(request,pk):
    #先判断有没有这个类
    category = get_object_or_404(Category,pk=pk)
    if category:
        post_list = category.post_set.filter(category=category)
        return render(request,'blog/index.html',context={'post_list':post_list})
    else:
        post_list = Post.objects.all()
        return render(request,'blog/index.html',context={
            'title':'我博客首页',
            'welcome':'欢迎访问我的博客首页',
            'post_list':post_list
        })

def allPostByCreateTime(request,year,month,day):  #这里进行查找的时候，必须是year,month,day
    post_list = Post.objects.filter(created_time__year=year,created_time__month=month,created_time__day=day)
    return render(request,'blog/index.html',context={'post_list':post_list})


def allPostByAuthorId(request,pk):
    post_list = Post.objects.filter(pk=pk)
    return render(request,'blog/index.html',context={'post_list':post_list})


def search(request):
    q = request.GET.get('q')
    error_msg = ''
    if not q:
        error_msg= '请输入关键字'
        return render(request,'blog/index.html',{'error_msg':error_msg})
    post_list = Post.objects.filter(Q(title__icontains=q)|Q(body__icontains=q))
    return render(request,'blog/index.html',context={
        'post_list':post_list,
        'error_msg':error_msg
    })

