# راهنمای ارزیابی مدل روی سرور بدون Unity

## 🎯 هدف
ارزیابی عملکرد مدل آموزش دیده روی داده‌های واقعی بدون نیاز به Unity

## 🚀 مراحل اجرا

### 1. آماده‌سازی محیط
```bash
# فعال کردن محیط مجازی
source .venv/bin/activate

# نصب وابستگی‌های اضافی
pip install matplotlib seaborn pandas
```

### 2. تست سریع (10 اپیزود)
```bash
# تست سریع با مدل موجود
python quick_test.py --model data/checkpoints/best_il_model.pth --episodes 10
```

### 3. ارزیابی کامل (50 اپیزود)
```bash
# ارزیابی کامل با تولید ویدیو
python evaluate_real_data.py --model data/checkpoints/best_il_model.pth --episodes 50 --render

# ارزیابی بدون ویدیو (سریع‌تر)
python evaluate_real_data.py --model data/checkpoints/best_il_model.pth --episodes 50
```

## 📊 نتایج قابل انتظار

### معیارهای موفقیت:
- **نرخ موفقیت**: درصد اپیزودهایی که 80%+ مایع و آلودگی را حذف کردند
- **کاهش مایع**: میانگین درصد کاهش مایع
- **کاهش آلودگی**: میانگین درصد کاهش آلودگی
- **پاداش**: میانگین پاداش دریافتی
- **ایمنی**: تعداد برخوردها و نقض‌های ایمنی

### خروجی‌ها:
- `evaluation_results/real_data_evaluation_YYYYMMDD_HHMMSS.json` - نتایج تفصیلی
- `evaluation_results/evaluation_summary_YYYYMMDD_HHMMSS.csv` - خلاصه نتایج
- `evaluation_results/evaluation_plots_YYYYMMDD_HHMMSS.png` - نمودارهای تحلیلی
- `evaluation_results/episode_XXX.mp4` - ویدیوهای اپیزودها (اگر render=True)

## 🔧 تنظیمات پیشرفته

### تغییر پارامترهای محیط:
```python
# در evaluate_real_data.py
self.env = MockSuctionEnv(
    image_size=(128, 128),
    max_steps=1000,
    reward_weights={
        "alpha": 2.0,      # وزن کاهش مایع
        "beta": 1.5,       # وزن کاهش آلودگی
        "lambda_time": -0.005,  # جریمه زمان
        "lambda_action": -0.0005,  # جریمه نرمی عمل
        "lambda_collision": -2.0,  # جریمه برخورد
        "lambda_safety": -5.0  # جریمه نقض ایمنی
    }
)
```

### تغییر معیارهای موفقیت:
```python
# در evaluate_real_data.py
def _is_success(self, info: Dict[str, Any]) -> bool:
    liquid_reduction = info.get('liquid_reduction', 0)
    contaminant_reduction = info.get('contaminant_reduction', 0)
    
    # معیارهای موفقیت قابل تنظیم
    return liquid_reduction > 0.8 and contaminant_reduction > 0.8
```

## 📈 تحلیل نتایج

### تفسیر معیارها:

1. **نرخ موفقیت > 80%**: عملکرد عالی
2. **نرخ موفقیت 60-80%**: عملکرد خوب
3. **نرخ موفقیت 40-60%**: عملکرد متوسط
4. **نرخ موفقیت < 40%**: نیاز به بهبود

### نمودارهای تحلیلی:
- **توزیع نرخ موفقیت**: نمایش تعداد اپیزودهای موفق/ناموفق
- **پراکندگی عملکرد**: رابطه بین کاهش مایع و آلودگی
- **توزیع پاداش**: توزیع پاداش‌های دریافتی
- **طول اپیزود vs عملکرد**: رابطه بین طول اپیزود و کیفیت عملکرد

## 🚨 عیب‌یابی

### مشکلات رایج:

1. **خطای بارگذاری مدل**:
```bash
# بررسی وجود فایل مدل
ls -la data/checkpoints/

# بررسی محتوای مدل
python -c "import torch; print(torch.load('data/checkpoints/best_il_model.pth').keys())"
```

2. **خطای CUDA**:
```bash
# استفاده از CPU
export CUDA_VISIBLE_DEVICES=""
python evaluate_real_data.py --model data/checkpoints/best_il_model.pth
```

3. **خطای حافظه**:
```bash
# کاهش تعداد اپیزودها
python evaluate_real_data.py --model data/checkpoints/best_il_model.pth --episodes 10
```

## 🎯 مثال کامل

```bash
# 1. تست سریع
python quick_test.py --model data/checkpoints/best_il_model.pth --episodes 5

# 2. ارزیابی کامل
python evaluate_real_data.py --model data/checkpoints/best_il_model.pth --episodes 50 --render

# 3. بررسی نتایج
ls -la evaluation_results/
cat evaluation_results/evaluation_summary_*.csv
```

## 📋 گزارش نمونه

```
📊 نتایج ارزیابی مدل روی داده‌های واقعی
============================================================
✅ نرخ موفقیت: 75.0%
📈 میانگین کاهش مایع: 68.5% ± 15.2%
📈 میانگین کاهش آلودگی: 72.1% ± 12.8%
💰 میانگین پاداش: 45.32 ± 8.67
⏱️ میانگین طول اپیزود: 456 ± 123
⚠️ میانگین برخورد: 2.1 ± 1.8
🛡️ میانگین نقض ایمنی: 0.3 ± 0.7
🎯 اپیزودهای عملکرد بالا: 38
📉 اپیزودهای عملکرد پایین: 5
============================================================
```

## 🔄 اجرای خودکار

### اسکریپت خودکار:
```bash
#!/bin/bash
# auto_evaluate.sh

echo "🚀 شروع ارزیابی خودکار..."

# تست سریع
python quick_test.py --model data/checkpoints/best_il_model.pth --episodes 10

# ارزیابی کامل
python evaluate_real_data.py --model data/checkpoints/best_il_model.pth --episodes 50

# نمایش نتایج
echo "📊 نتایج ارزیابی:"
ls -la evaluation_results/
```

### اجرا:
```bash
chmod +x auto_evaluate.sh
./auto_evaluate.sh
```

این راهنما به شما امکان ارزیابی کامل مدل روی داده‌های واقعی را بدون نیاز به Unity می‌دهد.
