library('ISLR') # набор данных Hitters
library('leaps') # функция regsubset() — отбор оптимального 
# подмножества переменных
library('glmnet') # функция glmnet() — лассо
library('pls') # регрессия на главные компоненты — pcr()
# и частный МНК — plsr()
library("MASS")
my.seed <- 1

data(Boston)
str(Boston)

fix(Boston)
names(Boston)

dim(Boston)

#Пошаговое включение
regfit.fwd <- regsubsets(crim ~ ., data = Boston,
                         nvmax = 13, method = 'forward')
summary(regfit.fwd)

round(coef(regfit.fwd, 7), 3)

# подгоняем модели с сочетаниями предикторов до 13 (максимум в данных)
regfit.full <- regsubsets(crim ~ ., Boston, nvmax = 13)
reg.summary <- summary(regfit.full)
reg.summary

# структура отчёта по модели (ищем характеристики качества)
names(reg.summary)

# R^2 и скорректированный R^2
round(reg.summary$rsq, 3)

# на графике
plot(1:13, reg.summary$rsq, type = 'b',
     xlab = 'Количество предикторов', ylab = 'R-квадрат')
# сода же добавим скорректированный R-квадрат
points(1:13, reg.summary$adjr2, col = 'red')
# модель с максимальным скорректированным R-квадратом
which.max(reg.summary$adjr2)

points(which.max(reg.summary$adjr2), 
       reg.summary$adjr2[which.max(reg.summary$adjr2)],
       col = 'red', cex = 2, pch = 20)
legend('bottomright', legend = c('R^2', 'R^2_adg'),
       col = c('black', 'red'), lty = c(1, NA),
       pch = c(1, 1))

# C_p
reg.summary$cp

# число предикторов у оптимального значения критерия
which.min(reg.summary$cp)

### 8
# график
plot(reg.summary$cp, xlab = 'Число предикторов',
     ylab = 'C_p', type = 'b')
points(which.min(reg.summary$cp),
       reg.summary$cp[which.min(reg.summary$cp)], 
       col = 'red', cex = 2, pch = 20)

# функция для прогноза для функции regsubset()
predict.regsubsets <- function(object, newdata, id, ...){
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  xvars <- names(coefi)
  mat[, xvars] %*% coefi
}

# k-кратная кросс-валидация
# отбираем 10 блоков наблюдений
k <- 10
set.seed(my.seed)
folds <- sample(1:k, nrow(Boston), replace = T)

# заготовка под матрицу с ошибками
cv.errors <- matrix(NA, k, 13, dimnames = list(NULL, paste(1:13)))

# заполняем матрицу в цикле по блокам данных
for (j in 1:k){
  best.fit <- regsubsets(crim ~ ., data = Boston[folds != j, ],
                         nvmax = 13)
  # теперь цикл по количеству объясняющих переменных
  for (i in 1:13){
    # модельные значения Boston
    pred <- predict(best.fit, Boston[folds == j, ], id = i)
    # вписываем ошибку в матрицу
    cv.errors[j, i] <- mean((Boston$crim[folds == j] - pred)^2)
  }
}

# усредняем матрицу по каждому столбцу (т.е. по блокам наблюдений), 
# чтобы получить оценку MSE для каждой модели с фиксированным 
# количеством объясняющих переменных
mean.cv.errors <- apply(cv.errors, 2, mean)
round(mean.cv.errors, 1)

# на графике
plot(mean.cv.errors, type = 'b')
points(which.min(mean.cv.errors), mean.cv.errors[which.min(mean.cv.errors)],
       col = 'red', pch = 20, cex = 2)

# перестраиваем модель с 12 объясняющими переменными на всём наборе данных
reg.best <- regsubsets(crim ~ ., data = Boston, nvmax = 13)
round(coef(reg.best, 12), 3)

#лассо-регрессия
# из-за синтаксиса glmnet() формируем явно матрицу объясняющих...
x <- model.matrix(crim ~ ., Boston)[, -1]

# и вектор значений зависимой переменной
y <- Boston$crim

set.seed(my.seed)
train <- sample(1:nrow(x), nrow(x)/2)
test <- -train
y.test <- y[test]

# вектор значений гиперпараметра лямбда
grid <- 10^seq(10, -2, length = 100)

lasso.mod <- glmnet(x[train, ], y[train], alpha = 1, lambda = grid)
plot(lasso.mod)

set.seed(my.seed)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1)
plot(cv.out)

bestlam <- cv.out$lambda.min
lasso.pred <- predict(lasso.mod, s = bestlam, newx = x[test, ])
#MSE на тестовой
round(mean((lasso.pred - y.test)^2), 0)

# коэффициенты лучшей модели
out <- glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef <- predict(out, type = 'coefficients',
                      s = bestlam)[1:13, ]
round(lasso.coef, 3)

round(lasso.coef[lasso.coef != 0], 3)
