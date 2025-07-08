from rest_framework import serializers
from .models import Option, OptionPrice, MarketData, TestingOptionData, ModelTrainingResult
from datetime import date

OPTION_TYPE_CHOICES = [
    ("C", "Call"),
    ("P", "Put"),
]

class OptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Option
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')


class OptionPriceSerializer(serializers.ModelSerializer):
    option = OptionSerializer(read_only=True)
    
    class Meta:
        model = OptionPrice
        fields = '__all__'
        read_only_fields = ('calculated_at',)


class MarketDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = MarketData
        fields = '__all__'
        read_only_fields = ('timestamp',)


class OptionCalculationSerializer(serializers.Serializer):
    """Serializer for option pricing calculations"""
    symbol = serializers.CharField(max_length=20)
    option_type = serializers.ChoiceField(choices=[('call', 'Call'), ('put', 'Put')])
    strike_price = serializers.DecimalField(max_digits=10, decimal_places=2)
    expiration_date = serializers.DateField()
    underlying_price = serializers.DecimalField(max_digits=10, decimal_places=2)
    risk_free_rate = serializers.DecimalField(max_digits=5, decimal_places=4, default=0.02)
    volatility = serializers.DecimalField(max_digits=5, decimal_places=4)
    
    def validate_expiration_date(self, value):
        from django.utils import timezone
        if value <= timezone.now().date():
            raise serializers.ValidationError("Expiration date must be in the future")
        return value
    
    def validate_strike_price(self, value):
        if value <= 0:
            raise serializers.ValidationError("Strike price must be positive")
        return value
    
    def validate_underlying_price(self, value):
        if value <= 0:
            raise serializers.ValidationError("Underlying price must be positive")
        return value
    
    def validate_volatility(self, value):
        if value <= 0 or value > 1:
            raise serializers.ValidationError("Volatility must be between 0 and 1")
        return value


class TestingOptionDataSerializer(serializers.ModelSerializer):
    """Serializer for TestingOptionData model"""
    option_type = serializers.ChoiceField(choices=OPTION_TYPE_CHOICES)
    days_to_expiry = serializers.SerializerMethodField()
    class Meta:
        model = TestingOptionData
        fields = '__all__'
        read_only_fields = ()

    def get_days_to_expiry(self, obj):
        if obj.trade_datetime and obj.expiration_date:
            return (obj.expiration_date - obj.trade_datetime.date()).days
        return None

    def validate_quantity(self, value):
        """Validate that quantity is not zero"""
        if value == 0:
            raise serializers.ValidationError("Quantity cannot be zero")
        return value

    def validate_trade_price(self, value):
        """Validate that trade price is positive"""
        if value < 0:
            raise serializers.ValidationError("Trade price must be positive")
        return value

    def validate_strike_price(self, value):
        """Validate that strike price is positive"""
        if value <= 0:
            raise serializers.ValidationError("Strike price must be positive")
        return value

    def to_representation(self, instance):
        """Custom representation to include computed moneyness"""
        data = super().to_representation(instance)
        # Add moneyness calculation if underlying price is available
        if hasattr(instance, 'close_price') and instance.close_price:
            data['moneyness'] = instance.compute_moneyness(instance.close_price)
        # Also include the stored moneyness value if it exists
        if instance.moneyness is not None:
            data['moneyness'] = instance.moneyness
        return data


class TestingOptionDataCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating TestingOptionData with validation"""
    class Meta:
        model = TestingOptionData
        fields = [
            'symbol', 'trade_datetime', 'quantity', 'trade_price', 
            'close_price', 'proceeds', 'comm_fee', 'basis', 
            'realized_pl', 'mtm_pl', 'code', 'underlying', 
            'strike_price', 'expiration_date', 'option_type'
        ]

    def validate(self, data):
        """Custom validation for the entire data set"""
        # Validate that expiration date is not in the past
        if data.get('expiration_date'):
            from django.utils import timezone
            if data['expiration_date'] < timezone.now().date():
                raise serializers.ValidationError("Expiration date cannot be in the past")
        
        # Validate that trade datetime is not in the future
        if data.get('trade_datetime'):
            from django.utils import timezone
            if data['trade_datetime'] > timezone.now():
                raise serializers.ValidationError("Trade datetime cannot be in the future")
        
        return data


class ModelTrainingResultSerializer(serializers.ModelSerializer):
    """Serializer for ModelTrainingResult model"""
    model_type_display = serializers.CharField(source='get_model_type_display', read_only=True)
    
    class Meta:
        model = ModelTrainingResult
        fields = '__all__'
        read_only_fields = ('training_date',)
