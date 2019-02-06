################################################################################
# Exercise build for WiMLDS workshop: MLE in tf
# Estimating a in Pythagorean theorem when b and c are given
################################################################################

# loading nessesary libraries
library(tidyverse)
library(tensorflow)

# Real data is: 
b <- 12
c <- 13
(a <- sqrt(c^2-b^2))
# The task is to estimate a, when b and c are given 

################################################################################
# MODEL
################################################################################

# Setting up nodes in graph
# Set b and c as constant value (name them tf_b, tf_c)
tf_b <- tf$constant(b)
tf_c <- tf$constant(c)

# Set a as variable (name it tf_a), initizalize its value as e.g. 10
tf_a <- tf$Variable(10)

# Set y_ as tf_a as it is what we estimate
c_ <- tf$sqrt(tf_b^2 + tf_a^2)

# To calculate loss function, we need to know what is a correct value of output
# let us define y as constant value

# We need to define how we want to compare y and y hat, so 
# we need to choose loss function or define it by ourselves 
# let us write a MSE loss function 
loss <- tf$losses$mean_squared_error(tf_c, c_)
# loss <- (y-y_)^2

# Before training, we need to decide which optimizer we want to use and 
optimizer <- tf$train$AdamOptimizer(learning_rate = 0.1)
# and that we want to minimize loss function
train <- optimizer$minimize(loss)
# train <- optimizer$minimize(loss_MSE)
################################################################################
# RUN MODEL 
################################################################################

# Let us upen a session
sess = tf$Session()

# let us initialize all variables
sess$run(tf$global_variables_initializer())

# Let us check what is a loss for randomly initialized value of a
sess$run(loss)
sess$run(tf_a)  

# Let us do one iteration of training
sess$run(train)

# Let us check the value of a parameter
sess$run(loss)
sess$run(tf_a)  

# Do this itaration once again or write a 'for' loop to see how a value changes 
training <- tibble(current_loss = sess$run(loss), 
                   current_value = sess$run(tf_a))
                   
for (i in 1:200){
  # train
  sess$run(train)
  # save results to training table
  training <- training %>% 
    rbind(tibble(current_loss = sess$run(loss), 
                 current_value = sess$run(tf_a)))
}

# Plotting the result 
training %>% 
  ggplot(aes(1:nrow(.), current_loss)) + 
  geom_line() + 
  scale_x_continuous('Step number') + 
  scale_y_continuous('Current loss')

training %>% 
  ggplot(aes(1:nrow(.), current_value)) + 
  geom_line() + 
  scale_x_continuous('Step number') + 
  scale_y_continuous('Current value')

training$current_value
training$current_loss
a
