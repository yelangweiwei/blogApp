"""
Django settings for blog0226 project.

Generated by 'django-admin startproject' using Django 1.11.4.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.11/ref/settings/
"""

import os
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.11/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '-=31i#)es5noymrqk@1_i@)0-*t!iamn@(t@+60ecn4+y&*)wj'

# SECURITY WA:RNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth', #用户认证系统
    'django.contrib.contenttypes',#是auth模块的用户权限处理部分依赖的应用
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'Blog.apps.BlogConfig',
    'comments.apps.CommentsConfig',
    'haystack',#添加高级搜索
    'userApp',#注册新建的应用
]

#添加高级搜索
HAYSTACK_CONNECTIONS={
    'default':{
        'ENGINE':'Blog.whoosh_cn_backend.WhooshEngine',
        'PATH':os.path.join(BASE_DIR,'whoolsh_index')
    }
}
HAYSTACK_SEARCH_RESULTS_PER_PAGE = 10
HAYSTACK_SIGNAL_PROCESSOR = 'haystack.signals.RealtimeSignalProcessor'

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware', #用于处理用户会话
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware', #绑定一个user对象到请求中
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

#让用户系统使用我们自定义的用户模型,使用profile不用
# AUTH_USER_MODEL = 'userApp.User'

ROOT_URLCONF = 'blogApp.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates'),
                 os.path.join(BASE_DIR, 'Blog/static')]
        ,
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'blogApp.wsgi.application'


# Database
# https://docs.djangoproject.com/en/1.11/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


# Password validation
# https://docs.djangoproject.com/en/1.11/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/1.11/topics/i18n/

#LANGUAGE_CODE = 'en-us'
#TIME_ZONE = 'UTC'

LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = 'Asia/Shanghai'

USE_I18N = True

USE_L10N = True

USE_TZ = True


#添加的配置
LOGOUT_REDIRECT_URL = '/'
LOGIN_REDIRECT_URL = '/'


#添加有右键的设置
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.qq.com'
EMAIL_PORT = 25
EMAIL_HOST_USER = '3485557481@qq.com'
EMAIL_HOST_PASSWORD = 'urywqqazigmlcjdb'
EMAIL_USER_TLS = True
EMAIL_FROM = '3485557481@qq.com'
DEFAULT_FROM_EMAIL = '3485557481@qq.com'


#编写验证的凭据,这个在登录的时候使用，第一个用来验证登录的用户名和密码是不是满足条件，第二个验证登录的邮箱地址是否满足条件
AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',
    'userApp.backends.EmailBackend',
)



# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.11/howto/static-files/

#static settings
STATIC_URL = '/static/'
#base_dir是项目的绝对地址
STATIC_ROOT = os.path.join(BASE_DIR,'collect_static')
#这个是各个静态文件和其他的静态文件
STATICFILES_DIRS=(os.path.join(BASE_DIR, 'common_static'),)
