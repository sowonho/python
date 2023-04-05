import numpy as np
import turtle
t = turtle.Turtle()
t.shape('circle')
t.shapesize(0.1)
t.speed(0)

xbegin = -200
t.goto(xbegin, 0)

t.fd(500)
t.bk(500)
t.left(90)
t.fd(300)
t.bk(300)
t.right(90)

t.color('red')
t.penup()

# 학습 데이터 생성
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 학습 데이터 정규화
X = (X - np.mean(X)) / np.std(X)

# 경사하강법 하이퍼파라미터 설정
learning_rate = 0.1
n_iterations = 400

# 모델 초기화
theta = np.random.randn(1, 2)
t.write(theta[0][0])

# 경사하강법 수행
for iteration in range(n_iterations):
    gradients = 2/100 * X.T.dot(X.dot(theta) - y)
    theta = theta - learning_rate * gradients
    ycor = float(theta[0][0])
    t.goto(xbegin + iteration, ycor * 100)
    t.stamp()
##    t.goto(iteration, 0)
##    t.fd(1)

# 학습된 모델 출력
t.write(theta)
