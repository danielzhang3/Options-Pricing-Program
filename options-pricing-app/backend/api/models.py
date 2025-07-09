from django.db import models
from django.utils import timezone
from datetime import date

OPTION_TYPE_CHOICES = [
    ("call", "Call"),
    ("put", "Put")
]


class Option(models.Model):
    OPTION_TYPES = [
        ('call', 'Call'),
        ('put', 'Put'),
    ]
    
    symbol = models.CharField(max_length=20)
    option_type = models.CharField(max_length=4, choices=OPTION_TYPES)
    strike_price = models.DecimalField(max_digits=10, decimal_places=2)
    expiration_date = models.DateField()
    underlying_price = models.DecimalField(max_digits=10, decimal_places=2)
    risk_free_rate = models.DecimalField(max_digits=5, decimal_places=4, default=0.02)
    volatility = models.DecimalField(max_digits=5, decimal_places=4)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['symbol', 'option_type', 'strike_price', 'expiration_date']
    
    def __str__(self):
        return f"{self.symbol} {self.option_type.upper()} {self.strike_price} {self.expiration_date}"


class OptionPrice(models.Model):
    option = models.ForeignKey(Option, on_delete=models.CASCADE, related_name='prices')
    price = models.DecimalField(max_digits=10, decimal_places=4)
    delta = models.DecimalField(max_digits=8, decimal_places=6, null=True, blank=True)
    gamma = models.DecimalField(max_digits=8, decimal_places=6, null=True, blank=True)
    theta = models.DecimalField(max_digits=8, decimal_places=6, null=True, blank=True)
    vega = models.DecimalField(max_digits=8, decimal_places=6, null=True, blank=True)
    rho = models.DecimalField(max_digits=8, decimal_places=6, null=True, blank=True)
    calculated_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-calculated_at']
    
    def __str__(self):
        return f"{self.option} - ${self.price}"


class MarketData(models.Model):
    symbol = models.CharField(max_length=20)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    volume = models.IntegerField()
    timestamp = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.symbol} - ${self.price} at {self.timestamp}"

class TestingOptionData(models.Model):
    symbol = models.CharField(max_length=100)
    trade_datetime = models.DateTimeField()
    quantity = models.IntegerField()
    trade_price = models.FloatField()  # t. price
    close_price = models.FloatField(null=True, blank=True)  # c. price
    proceeds = models.FloatField(null=True, blank=True)
    comm_fee = models.FloatField(null=True, blank=True)
    basis = models.FloatField(null=True, blank=True)
    realized_pl = models.FloatField(null=True, blank=True)
    mtm_pl = models.FloatField(null=True, blank=True)
    code = models.CharField(max_length=20, null=True, blank=True)

    underlying = models.CharField(max_length=20)
    strike_price = models.FloatField()
    expiration_date = models.DateField()
    option_type = models.CharField(max_length=4, choices=OPTION_TYPE_CHOICES)
    moneyness = models.FloatField(null=True, blank=True)  # optional: compute on save if needed

    def __str__(self):
        return f"{self.symbol} ({self.trade_datetime.strftime('%Y-%m-%d %H:%M')})"

    def compute_moneyness(self, underlying_price):
        if self.strike_price > 0:
            return underlying_price / self.strike_price
        return None

    def save(self, *args, **kwargs):
        if self.close_price and self.strike_price and self.strike_price > 0:
            self.moneyness = self.close_price / self.strike_price
        super().save(*args, **kwargs)

    def days_to_expiry(self):
        if self.trade_datetime and self.expiration_date:
            return (self.expiration_date - self.trade_datetime.date()).days
        return None

    class Meta:
        db_table = "testing_options_data"
        ordering = ["-trade_datetime"]


class ModelTrainingResult(models.Model):
    MODEL_TYPES = [
        ('linear_regression', 'Multiple Linear Regression'),
        ('neural_network', 'Neural Network'),
    ]
    
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    training_date = models.DateTimeField(default=timezone.now)
    
    alpha = models.FloatField(null=True, blank=True)  
    lambda_val = models.FloatField(null=True, blank=True)  
    num_iters = models.IntegerField(null=True, blank=True)  
    epochs = models.IntegerField(null=True, blank=True)  
    batch_size = models.IntegerField(null=True, blank=True)  
    validation_split = models.FloatField(null=True, blank=True)  
    
    mean_absolute_error = models.FloatField()
    mean_squared_error = models.FloatField()
    root_mean_squared_error = models.FloatField()
    r2_score = models.FloatField()
    mean_absolute_percentage_error = models.FloatField(null=True, blank=True)
    residual_std = models.FloatField()
    max_overprediction = models.FloatField()
    max_underprediction = models.FloatField()
    
    n_training_samples = models.IntegerField()
    n_test_samples = models.IntegerField()
    training_epochs = models.IntegerField(null=True, blank=True)  
    final_training_loss = models.FloatField(null=True, blank=True)  
    final_validation_loss = models.FloatField(null=True, blank=True)  
    
    coefficients = models.JSONField(null=True, blank=True)
    feature_names = models.JSONField(null=True, blank=True)
    
    training_inputs = models.JSONField(null=True, blank=True)
    
    class Meta:
        ordering = ['-training_date']
    
    def __str__(self):
        return f"{self.get_model_type_display()} - {self.training_date.strftime('%Y-%m-%d %H:%M')} - MAE: ${self.mean_absolute_error:.2f}"