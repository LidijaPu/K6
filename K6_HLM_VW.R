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
duom <- read.csv("vw.csv")                               # failas tame pačiame aplanke kaip kodas
duom$model <- trimws(duom$model)                         # pavadinimų priekyje yra tarpas

duom <- subset(duom, fuelType %in% c("Petrol", "Diesel"))   # tik benzininiai ir dyzeliniai
duom <- subset(duom, engineSize > 0)                        # variklio tūris 0 – klaida
duom <- subset(duom, mpg >= 20 & mpg <= 100)                # neįtikėtinos sąnaudų reikšmės
duom <- duom[duom$model %in% names(which(table(duom$model) >= 30)), ]  # modeliai su >= 30 automobilių
duom$model <- factor(duom$model)

duom$lkaina <- log(duom$price)        # priklausomas kintamasis – kainos logaritmas
duom$lkaina_c <- duom$lkaina - mean(duom$lkaina)   # centruota log(kaina) – besąlyginio modelio hipotezėms
duom$amzius <- 2020 - duom$year       # amžius metais (duomenys surinkti 2020 m.)
duom$rida   <- duom$mileage / 1000    # rida, tūkst. mylių
duom$vid_variklis <- ave(duom$engineSize, duom$model)   # II lygmens kintamasis: modelio vidutinis variklio tūris

duom$transmission <- factor(duom$transmission)   # Automatic, Manual, Semi-Auto
duom$fuelType <- factor(duom$fuelType)           # Diesel, Petrol
table(duom$transmission); table(duom$fuelType)

nrow(duom)            # stebėjimų skaičius
nlevels(duom$model)   # II lygmens aibių (modelių) skaičius
sort(table(duom$model))

# ---------------------------------------------------------------------
# 1. Priklausomo kintamojo normalumas
# ---------------------------------------------------------------------
ggplot(duom, aes(x = price)) +
  geom_histogram(bins = 30, fill = "#69b3a2", color = "white") +
  labs(title = "Priklausomo kintamojo histograma", x = "kaina", y = "Dažnis")
ggplot(duom, aes(x = lkaina)) +
  geom_histogram(bins = 30, fill = "#69b3a2", color = "white") +
  labs(title = "Priklausomo kintamojo histograma", x = "log(kaina)", y = "Dažnis")
set.seed(2026)
shapiro.test(sample(duom$lkaina, 5000))   # shapiro.test leidžia iki 5000 stebėjimų – atsitiktinė imtis

# ---------------------------------------------------------------------
# 2. Besąlyginis modelis ir ICC
# ---------------------------------------------------------------------
model.0 <- lme(lkaina_c ~ 1, random = ~ 1 | model, data = duom)   # REML, centruoti duomenys
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
#    I lygmuo: amzius, rida, mpg (intervaliniai), transmission, fuelType (kategoriniai)
#    II lygmuo: vid_variklis
#    Atsitiktinio poveikio: konstanta ir amzius
#    Kategoriniams kintamiesiems R automatiškai sukuria pseudokintamuosius
#    (bazinės kategorijos – Automatic ir Diesel)
# ---------------------------------------------------------------------
ctrl <- lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200)
model.1 <- lme(lkaina ~ 1 + vid_variklis + amzius + rida + mpg + transmission + fuelType,
               random = ~ 1 + amzius | model, data = duom, control = ctrl)   # REML
summary(model.1)        # fiksuoti parametrai (Fixed effects) ir st. nuokrypiai (Random effects)
VarCorr(model.1)        # dispersijos: tau00, tau11, sigma^2

model.0ml <- update(model.0, method = "ML")   # modelių palyginimui – ML įverčiai
model.1ml <- update(model.1, method = "ML")

# Ar reikalingas atsitiktinis amžiaus nuolydis?
model.1a <- lme(lkaina ~ 1 + vid_variklis + amzius + rida + mpg + transmission + fuelType,
                random = ~ 1 | model, data = duom, method = "ML", control = ctrl)
anova(model.1ml, model.1a)       # p < 0,0001 – atsitiktinis nuolydis reikalingas
AIC(model.0ml)
AIC(model.1ml)

# ---------------------------------------------------------------------
# 5. Modeliai su sąveika (II lygmens kintamasis × I lygmens kintamasis)
# ---------------------------------------------------------------------
# SĄVEIKA 1: vid_variklis ir amzius
# Klausimas: ar didesnio variklio modeliai nuvertėja greičiau?
model.2ml <- update(model.1ml, . ~ . + vid_variklis:amzius)
summary(model.2ml)$tTable
anova(model.1ml, model.2ml)      # sąveika nereikšminga – nereikalinga

# SĄVEIKA 2: vid_variklis ir rida
# Klausimas: ar didesnio variklio modeliams rida kainą veikia kitaip?
model.3ml <- update(model.1ml, . ~ . + vid_variklis:rida)
summary(model.3ml)$tTable
anova(model.1ml, model.3ml)      # sąveika nereikšminga – nereikalinga

# ---------------------------------------------------------------------
# 6. GALUTINIS MODELIS: dviejų lygmenų modelis be sąveikų (model.1, REML)
#    Abi sąveikos nereikšmingos, todėl renkamės paprastesnį modelį
# ---------------------------------------------------------------------
summary(model.1)       # fiksuoti parametrai ir st. nuokrypiai
VarCorr(model.1)       # dispersijos: tau00, tau11, sigma^2

# ---------------------------------------------------------------------
# 7. Galutinio modelio palyginimas su besąlyginiu
# ---------------------------------------------------------------------
sigma2_1 <- model.1$sigma^2
(sigma2_0 - sigma2_1) / sigma2_0              # I lygmens dispersijos sąlyginis pokytis
AIC(model.0ml, model.1ml)                     # anova() netinka – model.0 sudarytas centruotai priklausomai kintamajai
AIC(model.0ml) - AIC(model.1ml)               # Akaikės indekso pokytis

# ---------------------------------------------------------------------
# 8. Prielaidų tikrinimas: liekanų normalumas ir homoskedastiškumas
# ---------------------------------------------------------------------
qqnorm(model.1, ~ resid(.))      # I lygmens liekanos e
qqnorm(model.1, ~ ranef(.))      # II lygmens u0 ir u1
plot(model.1, resid(., type = "pearson") ~ fitted(.), abline = 0,
     main = "Liekanos vs prognozuojamos reikšmės")   # homoskedastiškumas
hist(resid(model.1), breaks = 80, main = "I lygmens liekanų histograma", xlab = "Liekanos")
mean(abs(resid(model.1, type = "pearson")) > 3)   # dalis liekanų, didesnių nei 3 st. nuokrypiai
round(ranef(model.1), 3)                          # kiekvieno modelio u0 ir u1

# ---------------------------------------------------------------------
# 9. Prognozė su galutiniu modeliu
#    3 metų VW Golf, 30 tūkst. mylių rida, 55 mpg, mechaninė dėžė, benzinas
# ---------------------------------------------------------------------
vv_golf <- unique(duom$vid_variklis[duom$model == "Golf"])   # Golf vidutinis variklio tūris
naujas_golf <- data.frame(
  model = factor("Golf", levels = levels(duom$model)),
  vid_variklis = vv_golf,
  amzius = 3,
  rida = 30,
  mpg = 55,
  transmission = factor("Manual", levels = levels(duom$transmission)),
  fuelType = factor("Petrol", levels = levels(duom$fuelType))
)

exp(predict(model.1, naujas_golf, level = 0))   # pagal fiksuotą dalį (level = 0), £
exp(predict(model.1, naujas_golf, level = 1))   # su Golf atsitiktiniais efektais (level = 1), £
