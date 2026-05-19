import joblib
import pandas as pd

model=joblib.load('house_price_rf_pipeline.pkl')

ALLOWED_RENOVATION = ["精装", "简装", "毛坯"]
ALLOWED_DISTRICTS = [
    "浦东", "闵行", "徐汇", "静安", "黄浦", "长宁",
    "普陀", "虹口", "杨浦", "宝山", "嘉定", "松江"
]
MIN_AREA = 10
MAX_AREA = 500
MIN_YEAR = 1980
MAX_YEAR = 2026
#输入校验
def safe_input(prompt, type_func, min_val=None, max_val=None):
    while True:
        try:
            value = type_func(input(prompt))
            if min_val is not None and value < min_val:
                print(f"❌ 输入不能小于 {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"❌ 输入不能大于 {max_val}")
                continue
            return value
        except ValueError:
            print("❌ 输入格式不对，请重新输入")

print("🏠 上海二手房价格预测小工具\n")

area = safe_input("面积（㎡）: ", float, MIN_AREA, MAX_AREA)
room = safe_input("卧室数量: ", int, 0, 10)
hall = safe_input("客厅数量: ", int, 0, 5)
toilet = safe_input("卫生间数量: ", int, 0, 5)
year = safe_input("建造年份: ", int, MIN_YEAR, MAX_YEAR)

# 字符串校验
while True:
    renovation = input("装修（精装 / 简装 / 毛坯）: ")
    if renovation in ALLOWED_RENOVATION:
        break
    print("❌ 请输入允许的装修类型")

while True:
    district = input("行政区: ")
    if district in ALLOWED_DISTRICTS:
        break
    print("❌ 请输入正确的行政区")
while True:
    floor_location = input("楼层位置（低楼层 / 中楼层 / 高楼层）: ")
    if floor_location in ["低楼层", "中楼层", "高楼层"]:
        break
    print("❌ 请输入正确的楼层位置")

input_df = pd.DataFrame([{
    "area": area,
    "room_count": room,
    "hall_count": hall,
    "toilet_count": toilet,
    "year_built": year,
    "renovation": renovation,
    "floor_location": floor_location,
    "district": district
    
}], columns=[
    "area",
    "room_count",
    "hall_count",
    "toilet_count",
    "year_built",
    "renovation",
    "floor_location", 
    "district"
])

pred = model.predict(input_df)[0]

print("\n📈 预测结果：")
print(f"预计总价：{pred:.2f} 万元")

