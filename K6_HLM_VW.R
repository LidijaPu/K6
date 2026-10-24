# =====================================================================
# Autorės: Lidija Puščiūtė, Darja Baranova
# K6(10) – Savarankiškas HLM: naudotų Volkswagen automobilių kainos
# Duomenys: Kaggle „100,000 UK Used Car Data set“ (failas vw.csv)
# I lygmuo – automobilis (skelbimas), II lygmuo – automobilio modelis
# =====================================================================

# Bibliotekos
library(nlme)      # lme() – hierarchiniai tiesiniai modeliai (kaip paskaitos pavyzdyje)
library(ggplot2)   # priklausomo kintamojo histograma

# ---------------------------------------------------------------------
# 0. Duomenų nuskaitymas ir paruošimas
# ---------------------------------------------------------------------
duom <- read.csv("vw.csv")
duom$model <- trimws(duom$model)                         # pavadinimų priekyje yra tarpas

duom <- subset(duom, fuelType %in% c("Petrol", "Diesel"))   # tik benzininiai ir dyzeliniai
duom <- subset(duom, engineSize > 0)                        # variklio tūris 0 – klaida
duom <- subset(duom, mpg >= 20 & mpg <= 100)                # neįtikėtinos sąnaudų reikšmės
duom <- duom[duom$model %in% names(which(table(duom$model) >= 30)), ]  # modeliai su >= 30 automobilių
duom$model <- factor(duom$model)

duom$lkaina <- log(duom$price)        # priklausomas kintamasis – kainos logaritmas
duom$amzius <- 2020 - duom$year       # amžius metais (duomenys surinkti 2020 m.)
duom$rida   <- duom$mileage / 1000    # rida, tūkst. mylių
duom$vid_variklis <- ave(duom$engineSize, duom$model)   # II lygmens kintamasis: modelio vidutinis variklio tūris

nrow(duom)            # stebėjimų skaičius
nlevels(duom$model)   # II lygmens aibių (modelių) skaičius
sort(table(duom$model))

# ---------------------------------------------------------------------
# 1. Priklausomo kintamojo normalumas
# ---------------------------------------------------------------------
ggplot(duom, aes(x = lkaina)) +
  geom_histogram(bins = 30, fill = "#69b3a2", color = "white") +
  labs(title = "Priklausomo kintamojo histograma", x = "log(kaina)", y = "Dažnis")
set.seed(2026)
shapiro.test(sample(duom$lkaina, 5000))   # shapiro.test leidžia iki 5000 stebėjimų – atsitiktinė imtis

# ---------------------------------------------------------------------
# 2. Besąlyginis modelis ir ICC
# ---------------------------------------------------------------------
model.0 <- lme(lkaina ~ 1, random = ~ 1 | model, data = duom)   # REML
summary(model.0)
VarCorr(model.0)
tau00 <- as.numeric(VarCorr(model.0)[1, "Variance"])
sigma2_0 <- model.0$sigma^2
ICC <- tau00 / (tau00 + sigma2_0)
ICC
intervals(model.0)    # PI dispersijų st. nuokrypiams (ar tau00 > 0, sigma^2 > 0)

# ---------------------------------------------------------------------
# 3. Intervalinių I lygmens kintamųjų koreliacijos
# ---------------------------------------------------------------------
round(cor(duom[, c("amzius", "rida", "mpg")]), 3)

# ---------------------------------------------------------------------
# 4. Dviejų lygmenų modelis
#    I lygmuo: amzius, rida, mpg;  II lygmuo: vid_variklis
#    Atsitiktinio poveikio: konstanta ir amzius
# ---------------------------------------------------------------------
ctrl <- lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200)
model.1 <- lme(lkaina ~ 1 + vid_variklis + amzius + rida + mpg,
               random = ~ 1 + amzius | model, data = duom, control = ctrl)   # REML
summary(model.1)        # fiksuoti parametrai (Fixed effects) ir st. nuokrypiai (Random effects)
VarCorr(model.1)        # dispersijos: tau00, tau11, sigma^2

# ---------------------------------------------------------------------
# 5. Modelių palyginimas su besąlyginiu
# ---------------------------------------------------------------------
sigma2_1 <- model.1$sigma^2
(sigma2_0 - sigma2_1) / sigma2_0      # I lygmens dispersijos sąlyginis pokytis

model.0ml <- update(model.0, method = "ML")   # palyginimui – ML įverčiai
model.1ml <- update(model.1, method = "ML")
anova(model.1ml, model.0ml)                   # AIC ir tikėtinumo santykio testas
AIC(model.0ml) - AIC(model.1ml)               # Akaikės indekso pokytis

# Ar reikalingas atsitiktinis amžiaus nuolydis?
model.1a <- lme(lkaina ~ 1 + vid_variklis + amzius + rida + mpg,
                random = ~ 1 | model, data = duom, method = "ML")
anova(model.1ml, model.1a)

# ---------------------------------------------------------------------
# 6. Modelis su sąveika: ar modelio variklis keičia nuvertėjimo greitį
#    beta1 = gamma10 + gamma11 * vid_variklis + u1
# ---------------------------------------------------------------------
model.2ml <- lme(lkaina ~ 1 + vid_variklis + amzius + rida + mpg + vid_variklis:amzius,
                 random = ~ 1 + amzius | model, data = duom, method = "ML", control = ctrl)
summary(model.2ml)$tTable
anova(model.1ml, model.2ml)

# ---------------------------------------------------------------------
# 7. Prielaidų tikrinimas – liekanų ir atsitiktinių efektų q-q grafikai
# ---------------------------------------------------------------------
qqnorm(model.1, ~ resid(.))      # I lygmens liekanos e
qqnorm(model.1, ~ ranef(.))      # II lygmens u0 ir u1
hist(resid(model.1), breaks = 80, main = "I lygmens liekanų histograma", xlab = "Liekanos")
mean(abs(resid(model.1, type = "pearson")) > 3)   # dalis liekanų, didesnių nei 3 st. nuokrypiai
round(ranef(model.1), 3)

# ---------------------------------------------------------------------
# 8. Prognozė: 3 metų Golf, 30 tūkst. mylių, 55 mpg
# ---------------------------------------------------------------------
vv_golf <- unique(duom$vid_variklis[duom$model == "Golf"])
naujas <- data.frame(model = "Golf", vid_variklis = vv_golf, amzius = 3, rida = 30, mpg = 55)
exp(predict(model.1, naujas, level = 0))   # pagal fiksuotą dalį („vidutinis“ modelis)
exp(predict(model.1, naujas, level = 1))   # įvertinus Golf atsitiktinius efektus
