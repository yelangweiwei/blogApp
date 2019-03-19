from django.shortcuts import render,get_object_or_404,redirect
from comments.models import Comments
from comments.forms import CommentsForm
from django.utils.six import python_2_unicode_compatible
from Blog.models import Post

# Create your views here.
@python_2_unicode_compatible
def post_comment(request,post_pk):
    post = get_object_or_404(Post,pk=post_pk)
    if request.method=='POST':
        form = CommentsForm(request.POST)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.post = post
            comment.save()
            return redirect(post)
        else:
            comment_list = post.comments_set.all()
            comment_count = post.comments_set.count()
            context = { 'post':post,
                        'form':form,
                        'comment_list':comment_list,
                        'comment_count':comment_count
                        }
            return render(request,'blog/detail.html',context=context)
    else:
        return redirect(post)





