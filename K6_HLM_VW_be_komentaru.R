library(nlme)
library(ggplot2)

duom <- read.csv("vw.csv")
duom$model <- trimws(duom$model)

duom <- subset(duom, fuelType %in% c("Petrol", "Diesel"))
duom <- subset(duom, engineSize > 0)
duom <- subset(duom, mpg >= 20 & mpg <= 100)
duom <- duom[duom$model %in% names(which(table(duom$model) >= 30)), ]
duom$model <- factor(duom$model)

duom$lkaina <- log(duom$price)
duom$lkaina_c <- duom$lkaina - mean(duom$lkaina)
duom$amzius <- 2020 - duom$year
duom$rida   <- duom$mileage / 1000
duom$vid_variklis <- ave(duom$engineSize, duom$model)

duom$transmission <- factor(duom$transmission)
duom$fuelType <- factor(duom$fuelType)
table(duom$transmission); table(duom$fuelType)

nrow(duom)
nlevels(duom$model)
sort(table(duom$model))

ggplot(duom, aes(x = price)) +
  geom_histogram(bins = 30, fill = "#69b3a2", color = "white") +
  labs(title = "Priklausomo kintamojo histograma", x = "kaina", y = "Dažnis")
ggplot(duom, aes(x = lkaina)) +
  geom_histogram(bins = 30, fill = "#69b3a2", color = "white") +
  labs(title = "Priklausomo kintamojo histograma", x = "log(kaina)", y = "Dažnis")
set.seed(2026)
shapiro.test(sample(duom$lkaina, 5000))

model.0 <- lme(lkaina_c ~ 1, random = ~ 1 | model, data = duom)
summary(model.0)
VarCorr(model.0)
tau00 <- as.numeric(VarCorr(model.0)[1, "Variance"])
sigma2_0 <- model.0$sigma^2
ICC <- tau00 / (tau00 + sigma2_0)
ICC
intervals(model.0)

round(cor(duom[, c("amzius", "rida", "mpg")]), 3)

ctrl <- lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200)
model.1 <- lme(lkaina ~ 1 + vid_variklis + amzius + rida + mpg + transmission + fuelType,
               random = ~ 1 + amzius | model, data = duom, control = ctrl)
summary(model.1)
VarCorr(model.1)

model.0ml <- update(model.0, method = "ML")
model.1ml <- update(model.1, method = "ML")

model.1a <- lme(lkaina ~ 1 + vid_variklis + amzius + rida + mpg + transmission + fuelType,
                random = ~ 1 | model, data = duom, method = "ML", control = ctrl)
anova(model.1ml, model.1a)
AIC(model.0ml)
AIC(model.1ml)

model.2ml <- update(model.1ml, . ~ . + vid_variklis:amzius)
summary(model.2ml)$tTable
anova(model.1ml, model.2ml)

model.3ml <- update(model.1ml, . ~ . + vid_variklis:rida)
summary(model.3ml)$tTable
anova(model.1ml, model.3ml)

summary(model.1)
VarCorr(model.1)

sigma2_1 <- model.1$sigma^2
(sigma2_0 - sigma2_1) / sigma2_0
AIC(model.0ml, model.1ml)
AIC(model.0ml) - AIC(model.1ml)

qqnorm(model.1, ~ resid(.))
qqnorm(model.1, ~ ranef(.))
plot(model.1, resid(., type = "pearson") ~ fitted(.), abline = 0,
     main = "Liekanos vs prognozuojamos reikšmės")
hist(resid(model.1), breaks = 80, main = "I lygmens liekanų histograma", xlab = "Liekanos")
mean(abs(resid(model.1, type = "pearson")) > 3)
round(ranef(model.1), 3)

vv_golf <- unique(duom$vid_variklis[duom$model == "Golf"])
naujas_golf <- data.frame(
  model = factor("Golf", levels = levels(duom$model)),
  vid_variklis = vv_golf,
  amzius = 3,
  rida = 30,
  mpg = 55,
  transmission = factor("Manual", levels = levels(duom$transmission)),
  fuelType = factor("Petrol", levels = levels(duom$fuelType))
)

exp(predict(model.1, naujas_golf, level = 0))
exp(predict(model.1, naujas_golf, level = 1))
