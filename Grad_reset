Perfect 👍
Let’s do exactly what you asked:

👉 Perform **two derivatives one after another**,
👉 Use **reset in between**,
👉 Observe how the flow works step-by-step.

We’ll also print everything so you can SEE what is happening.

---

# 🧪 Full Demonstration Code (Run in Colab)

```python
import torch

# Step 1: Create tensor
x = torch.arange(4.0, requires_grad=True)
print("Initial x:", x)

# -------------------------------
# FIRST DERIVATIVE
# -------------------------------
y1 = 2 * torch.dot(x, x)   # y1 = 2 * sum(x^2)
print("\ny1 value:", y1.item())

y1.backward()
print("Gradient after first backward:")
print("x.grad =", x.grad)

# -------------------------------
# RESET GRADIENT
# -------------------------------
x.grad.zero_()
print("\nAfter reset:")
print("x.grad =", x.grad)

# -------------------------------
# SECOND DERIVATIVE
# -------------------------------
y2 = x.sum()
print("\ny2 value:", y2.item())

y2.backward()
print("Gradient after second backward:")
print("x.grad =", x.grad)
```

---

# 🧠 Now let’s walk through what happens

---

## ✅ Step 1 — Initial x

```python
x = [0., 1., 2., 3.]
```

---

---

## 🔵 First computation

```python
y1 = 2 * torch.dot(x, x)
```

Mathematically:

[
y_1 = 2(x_0^2 + x_1^2 + x_2^2 + x_3^2)
]

Value:

```
y1 = 28
```

---

### 🔙 First backward

```python
y1.backward()
```

Derivative:

[
\frac{dy_1}{dx_i} = 4x_i
]

So:

```
x.grad = [0, 4, 8, 12]
```

---

---

## ♻️ Reset gradients

```python
x.grad.zero_()
```

Now:

```
x.grad = [0, 0, 0, 0]
```

⚠️ Only gradient memory cleared.
`x` is still:

```
[0., 1., 2., 3.]
```

---

---

## 🟢 Second computation

```python
y2 = x.sum()
```

Mathematically:

[
y_2 = x_0 + x_1 + x_2 + x_3
]

Value:

```
y2 = 6
```

---

### 🔙 Second backward

```python
y2.backward()
```

Derivative:

[
\frac{dy_2}{dx_i} = 1
]

So:

```
x.grad = [1, 1, 1, 1]
```

---

---

# 🎯 Final Output Summary

| Stage              | x value   | x.grad     |
| ------------------ | --------- | ---------- |
| Initial            | [0,1,2,3] | None       |
| After 1st backward | [0,1,2,3] | [0,4,8,12] |
| After reset        | [0,1,2,3] | [0,0,0,0]  |
| After 2nd backward | [0,1,2,3] | [1,1,1,1]  |

---

---

# 🧠 Core Understanding

✔️ Each forward pass creates a new computation graph
✔️ `.backward()` computes gradient for that graph
✔️ `.zero_()` clears only old gradients
✔️ x values never change unless you modify them
✔️ Gradients always reflect latest backward

---

---
