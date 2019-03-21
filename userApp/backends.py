from django.contrib.auth.models import User


class EmailBackend(object):
    def authenticate(self,request,**credentials):
        #注意：登录表单中用户名或者邮箱的field名均为username
        email = credentials.get('email',credentials.get('username'))
        try:
            user = User.objects.get(email=email)[0]
        except User.DoesNotExist:
            pass
        else:
            if user.check_password(credentials['password']):
                return user

    def get_user(self,user_id):
        '''
        这个必有
        :param user_id: 
        :return: 
        '''
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
