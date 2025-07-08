from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'options', views.OptionViewSet)
router.register(r'option-prices', views.OptionPriceViewSet)
router.register(r'market-data', views.MarketDataViewSet)
router.register(r'testing-options', views.TestingOptionDataViewSet)
router.register(r'training-results', views.ModelTrainingResultViewSet)

urlpatterns = [
    path('', include(router.urls)),

    path('predict/', views.predict_option_price, name='predict-option-price'),
    path('options/<int:option_id>/history/', views.option_price_history, name='option-price-history'),
    path('market-data/<str:symbol>/latest/', views.market_data_latest, name='market-data-latest'),
    path('upload-csv/', views.CSVUploadView.as_view(), name='upload-csv'),
    path('train-regression/', views.train_regression_model, name='train-regression'),
    path('train-neural-network/', views.train_neural_network_model, name='train-neural-network'),
    path('black-scholes-batch/', views.black_scholes_batch, name='black-scholes-batch'),
]
