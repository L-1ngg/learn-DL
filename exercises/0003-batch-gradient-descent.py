examples = [
    (1, 2),
    (2, 4),
    (3, 6),
]

w = 1.0
learning_rate = 0.05

for step in range(5):
    total_loss = 0.0
    total_gradient = 0.0

    # 本轮所有例子使用同一个尚未更新的 w。
    for x, target in examples:
        prediction = x * w
        loss = (prediction - target) ** 2
        gradient = 2 * x * (prediction - target)

        total_loss += loss
        total_gradient += gradient

    mean_loss = total_loss / len(examples)
    mean_gradient = total_gradient / len(examples)

    # 打印更新之前的状态，然后统一更新一次。
    print("step:", step, "w:", w, "mean_loss:", mean_loss)
    w = w - learning_rate * mean_gradient

print("final_w:", w)

# 训练已经结束，固定参数，对新输入进行推理。
# 本课人为构造的规则是目标 = 2 * 输入，因此这里已知目标为 8。
new_x = 4
new_target = 8
new_prediction = new_x * w
print("new_x:", new_x, "prediction:", new_prediction, "target:", new_target)
