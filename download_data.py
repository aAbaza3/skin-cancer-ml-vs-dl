import os
import zipfile

# تأكدي إن مكتبة kaggle مثبّتة
try:
    import kaggle
except ImportError:
    os.system("pip install kaggle")

# تحميل الداتا باستخدام مكتبة kaggle
# مكتبة kaggle تستخدم الملف الموجود في C:\Users\<اسمك>\.kaggle\kaggle.json

# اسم الداتا
dataset = "kmader/skin-cancer-mnist-ham10000"
output_dir = "data"

# إنشاء مجلد data لو مش موجود
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# تحميل البيانات
print("Downloading dataset...")
os.system(f"kaggle datasets download -d {dataset} -p {output_dir}")

# فك الضغط
print("Unzipping files...")
for file in os.listdir(output_dir):
    if file.endswith(".zip"):
        with zipfile.ZipFile(os.path.join(output_dir, file), 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"✅ Extracted {file}")
