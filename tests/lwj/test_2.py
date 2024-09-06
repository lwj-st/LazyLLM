import random

def test_random_number():
    random_number = random.randint(1, 10)  # 生成1到10之间的随机数
    print(f"Generated random number: {random_number}")
    assert random_number >= 5, f"Test failed: The random number {random_number} is less than 5"

