from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('admin-dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('prediction/', views.prediction_view, name='prediction'),
    path('result/<int:record_id>/', views.result_view, name='result'),
    path('history/', views.history_view, name='history'),
    path('delete-record/<int:record_id>/', views.delete_record, name='delete_record'),
]
