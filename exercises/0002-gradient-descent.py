# 一个训练例子：输入是 3，目标值是 6。
x = 3
target = 6
w = 1.0
learning_rate = 0.05

for step in range(5):
    prediction = x * w
    loss = (prediction - target) ** 2
    gradient = 2 * x * (prediction - target)

    # 打印的是本轮更新之前的参数和平方误差。
    print("step:", step, "w:", w, "loss:", loss)

    w = w - learning_rate * gradient
