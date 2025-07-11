# Generated by Django 5.2.3 on 2025-07-01 18:18

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MarketData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('symbol', models.CharField(max_length=20)),
                ('price', models.DecimalField(decimal_places=2, max_digits=10)),
                ('volume', models.IntegerField()),
                ('timestamp', models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                'ordering': ['-timestamp'],
            },
        ),
        migrations.CreateModel(
            name='Option',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('symbol', models.CharField(max_length=20)),
                ('option_type', models.CharField(choices=[('call', 'Call'), ('put', 'Put')], max_length=4)),
                ('strike_price', models.DecimalField(decimal_places=2, max_digits=10)),
                ('expiration_date', models.DateField()),
                ('underlying_price', models.DecimalField(decimal_places=2, max_digits=10)),
                ('risk_free_rate', models.DecimalField(decimal_places=4, default=0.02, max_digits=5)),
                ('volatility', models.DecimalField(decimal_places=4, max_digits=5)),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'unique_together': {('symbol', 'option_type', 'strike_price', 'expiration_date')},
            },
        ),
        migrations.CreateModel(
            name='OptionPrice',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('price', models.DecimalField(decimal_places=4, max_digits=10)),
                ('delta', models.DecimalField(blank=True, decimal_places=6, max_digits=8, null=True)),
                ('gamma', models.DecimalField(blank=True, decimal_places=6, max_digits=8, null=True)),
                ('theta', models.DecimalField(blank=True, decimal_places=6, max_digits=8, null=True)),
                ('vega', models.DecimalField(blank=True, decimal_places=6, max_digits=8, null=True)),
                ('rho', models.DecimalField(blank=True, decimal_places=6, max_digits=8, null=True)),
                ('calculated_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('option', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='prices', to='api.option')),
            ],
            options={
                'ordering': ['-calculated_at'],
            },
        ),
    ]
