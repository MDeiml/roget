#!/usr/bin/Rscript

data <- read.table('entropy.dat', col.names = c("entropy", "guesses"))

library(ggplot2)

lin <- glm(guesses ~ entropy, data = data)
exp <- glm(exp(guesses) ~ entropy, data = data)
log <- glm(log(guesses) ~ entropy, data = data)
inv <- glm((1/guesses) ~ entropy, data = data)
sqr <- glm(sqrt(guesses) ~ entropy, data = data)
pow <- glm(guesses ^ 2 ~ entropy, data = data)

print(lin)
print(exp)
print(log)
print(inv)
print(sqr)
print(pow)

p <- ggplot(data, aes(entropy, guesses)) + geom_jitter(alpha = 0.2, size = 0.7, width = 0.1, height = 0.2)
xs <- seq(0,8,length=36)
lin_pred <- predict(lin, newdata = data.frame(entropy = xs))
lin_pred <- data.frame(entropy = xs, guesses = lin_pred)
exp_pred <- predict(exp, newdata = data.frame(entropy = xs))
exp_pred <- log(exp_pred)
exp_pred <- data.frame(entropy = xs, guesses = exp_pred)
log_pred <- predict(log, newdata = data.frame(entropy = xs))
log_pred <- exp(log_pred)
log_pred <- data.frame(entropy = xs, guesses = log_pred)
inv_pred <- predict(inv, newdata = data.frame(entropy = xs))
inv_pred <- 1/inv_pred
inv_pred <- data.frame(entropy = xs, guesses = inv_pred)
sqr_pred <- predict(sqr, newdata = data.frame(entropy = xs))
sqr_pred <- sqr_pred ^ 2
sqr_pred <- data.frame(entropy = xs, guesses = sqr_pred)
pow_pred <- predict(pow, newdata = data.frame(entropy = xs))
pow_pred <- sqrt(pow_pred)
pow_pred <- data.frame(entropy = xs, guesses = pow_pred)
#also <- log(xs * 3.996 + 4.121)
#also <- data.frame(entropy = xs, guesses = also)
#print(pred)
p <- p + geom_line(data = lin_pred, aes(entropy, guesses), color = "red")
p <- p + geom_line(data = exp_pred, aes(entropy, guesses), color = "green")
p <- p + geom_line(data = log_pred, aes(entropy, guesses), color = "purple")
p <- p + geom_line(data = inv_pred, aes(entropy, guesses), color = "blue")
p <- p + geom_line(data = sqr_pred, aes(entropy, guesses), color = "orange")
p <- p + geom_line(data = pow_pred, aes(entropy, guesses), color = "pink")
p <- p + ylim(0, 10)
ggsave("plot.png", p)