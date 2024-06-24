
将glass_train$Type列纵向并入新列中！指定引入的新列的名Type！
```R
glass_train <- cbind(predict(glass_pp, glass_train[1:9]), Type = glass_train$Type)
```