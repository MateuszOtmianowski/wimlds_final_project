################################################################################
# Exercise build for WiMLDS workshop: MLE in tf
# Estimating factors: a and b, based on multiplication table 
################################################################################

# loading nessesary libraries
library(tidyverse)
library(tensorflow)

# Let us prepare some data for multiplication table with dim 10x10
data_multiplication_table <- expand.grid(a_value = 1:10, 
                                         b_value = 1:10) %>% 
    mutate(product = a_value*b_value, 
           a_id = as.factor(a_value), 
           b_id = as.factor(b_value)) %>% 
    as.tibble() 

# Preparing data to use in tensorflow
# our indicies need to be integers
# it works in python so we need indicies from 0
tf_data <- data_multiplication_table %>% 
    mutate(a_id_0 = as.integer(as.integer(a_id) - 1),
           b_id_0 = as.integer(as.integer(b_id) - 1))

################################################################################
# MODEL
################################################################################

# Let us define our constant values - a_indices and b_indicies
a_indices <- tf$constant(tf_data$a_id_0, dtype = 'int32')
b_indices <- tf$constant(tf_data$b_id_0, dtype = 'int32')

# Let us define our variables which we'll be estimating and updating in each iteration
# let us assume that variables are uniformly distributed with minimal value equals 1 and maximal value equals 10 

t_a <- tf$Variable(
    tf$random_uniform(
        shape = shape(length(levels(tf_data$a_id))),
        minval = 1, 
        maxval = 10
    ),
    name = 'a'  
)

t_b <- tf$Variable(
    tf$random_uniform(
        shape = shape(length(levels(tf_data$b_id))),
        minval = 1,
        maxval = 10
    ),
    name = 'b'  
)

# Now we need to combined variables to their indicies with gather function
t_a_gathered <- tf$gather(t_a, a_indices)
t_b_gathered <- tf$gather(t_b, b_indices)

# Nodes a & b are ready, but we need to define our output, so y hat
# y hat is a result of multiplying a and b 
y_ <- t_a_gathered * t_b_gathered

# To calculate loss function, we need to know what is a correct value of output
# let us define y as constant value
y <- tf$constant(tf_data$product)

# We need to define how we want to compare y and y hat, so 
# we need to choose loss function
loss <- tf$losses$mean_squared_error(y, y_)

# Before training, we need to decide which optimizer we want to use and 
optimizer <- tf$train$AdamOptimizer(learning_rate=0.1)
# and that we want to minimize loss function
train <- optimizer$minimize(loss)

# Let us combine t_a_gathered, t_b_gathered, y_, loss, optimizer and train into one list called model
model <- list(
    t_a_gathered = t_a_gathered,
    t_b_gathered = t_b_gathered,
    y_ = y_,
    loss = loss, 
    optimizer = optimizer, 
    train = train
)

################################################################################
# RUN MODEL 
################################################################################

# Let us upen a session
sess = tf$Session()

# let us initialize all variables
sess$run(tf$global_variables_initializer())

# let us take a look at the initialized values of t_a_gathered, t_b_gathered and y_
sess$run(t_a_gathered)
sess$run(t_b_gathered)
sess$run(y_)

# Let us check what is a loss for randomly initialized values
sess$run(model$loss)

# Let us prepare a helper table losses with step number and loss value
step = 0
losses = tibble(
    step = step,
    current_loss = sess$run(model$loss)
)

# Let us prepare a helper table params with step number, values of parameters a in a given estimation step, 
# values of parameters b in a given estimation step, and indicies 1:10 for each value (just to plot it later)
params <- tibble(step = step,
                 a_calc = sess$run(t_a),
                 a_ind = 1:10, 
                 b_calc = sess$run(t_b), 
                 b_ind = 1:10)

# TRAIN your model and save results to helper tables (losses, params)
# do it in for loop 
# use at least 250 steps

for (i in 1:500) {
    
    sess$run(model$train)
    
    losses = rbind(
        losses,
        tibble(step = step + i,
               current_loss = sess$run(model$loss))
    )
    
    params = rbind(
        params, 
        tibble(step = step + i,
               a_calc = sess$run(t_a), 
               a_ind = 1:10, 
               b_calc = sess$run(t_b),
               b_ind = 1:10)
    )
    
    cat(
        sprintf(
            "Step number: %1f\tLoss: %f\n", i, sess$run(model$loss)))
}
    
# Let's plot our results
losses
params %>% tail(15)

losses %>% 
    # filter(step > 250) %>% 
    ggplot(aes(step, current_loss)) + 
    geom_line()

params %>% 
  # filter(step < 250) %>% 
  ggplot(aes(step, a_calc)) + 
  geom_line() + 
  facet_grid(~a_ind) + 
  scale_x_continuous(name = 'Step number') +
  scale_y_continuous('Estimation of a value', breaks = 1:10) + 
  theme(text = element_text())
    
params %>% 
  filter(step < 250) %>% 
  ggplot(aes(step, b_calc)) + 
  geom_line() + 
  facet_grid(~b_ind) + 
  scale_x_continuous(name = 'Step number') +
  scale_y_continuous('Estimation of b value', breaks = 1:10)
