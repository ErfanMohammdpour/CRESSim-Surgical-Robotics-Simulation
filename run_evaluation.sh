#!/bin/bash

# اسکریپت ارزیابی خودکار مدل روی سرور
# Automatic Model Evaluation Script for Server

echo "🚀 شروع ارزیابی مدل روی داده‌های واقعی..."
echo "================================================"

# بررسی وجود محیط مجازی
if [ ! -d ".venv" ]; then
    echo "❌ محیط مجازی یافت نشد. لطفاً ابتدا محیط را ایجاد کنید."
    exit 1
fi

# فعال کردن محیط مجازی
echo "📱 فعال کردن محیط مجازی..."
source .venv/bin/activate

# بررسی وجود مدل
MODEL_PATH="data/checkpoints/best_il_model.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ فایل مدل یافت نشد: $MODEL_PATH"
    echo "💡 ابتدا مدل را آموزش دهید:"
    echo "   python manage.py train-il"
    exit 1
fi

# ایجاد پوشه نتایج
mkdir -p evaluation_results

echo "✅ محیط آماده است"
echo ""

# تست سریع (5 اپیزود)
echo "🔍 تست سریع (5 اپیزود)..."
python quick_test.py --model "$MODEL_PATH" --episodes 5

echo ""
echo "================================================"

# ارزیابی کامل (20 اپیزود)
echo "📊 ارزیابی کامل (20 اپیزود)..."
python evaluate_real_data.py --model "$MODEL_PATH" --episodes 20 --render

echo ""
echo "================================================"

# نمایش نتایج
echo "📋 نتایج ارزیابی:"
echo ""

if [ -d "evaluation_results" ]; then
    echo "📁 فایل‌های تولید شده:"
    ls -la evaluation_results/
    
    echo ""
    echo "📊 خلاصه نتایج:"
    if [ -f evaluation_results/evaluation_summary_*.csv ]; then
        cat evaluation_results/evaluation_summary_*.csv
    fi
else
    echo "❌ پوشه نتایج ایجاد نشد"
fi

echo ""
echo "✅ ارزیابی کامل شد!"
echo "📁 نتایج در پوشه evaluation_results/ ذخیره شدند"
