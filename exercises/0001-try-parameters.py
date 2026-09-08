x = 3
target = 6

best_w = 0
best_loss = abs(x * best_w - target)

for w in [1, 2, 3]:
    prediction = x * w
    loss = abs(prediction - target)
    print("w =", w, "prediction =", prediction, "loss =", loss)

    if loss < best_loss:
        best_w = w
        best_loss = loss

print("best_w =", best_w, "best_loss =", best_loss)
