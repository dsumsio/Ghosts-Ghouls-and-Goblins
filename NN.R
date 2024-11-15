library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(discrim)
library(themis)

sample = vroom('sample_submission.csv')
train_data = vroom('train.csv')
test_data = vroom('test.csv')
test_data$color = as.factor(test_data$color)


nn_recipe <- recipe(type ~ ., data=train_data) %>%
  update_role(id, new_role="id") %>%
  step_mutate_at(color, fn = factor) %>% 
  step_dummy(color) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_zv(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_numeric_predictors(), threshold=0.8) %>%
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 250 #or 100 or 250
              ) %>%
  set_engine("keras") %>% #verbose = 0 prints off less
  set_mode("classification")

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 25)),
                            levels=1)

folds <- vfold_cv(train_data, v = 5, repeats = 1)

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

tuned_nn <- nn_wf %>%
    tune_grid(resamples = folds,
              grid = nn_tuneGrid,
              metrics = metric_set(roc_auc))

tuned_nn %>% collect_metrics() %>%
    filter(.metric=="accuracy") %>%
    ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## Find Best Tuning Parameters
bestTune <- tuned_nn %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

preds <- final_wf %>%
  predict(new_data = test_data, type = 'class')

## Format the Predictions for Submission to Kaggle
kaggle_submission <-  preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(id, .pred_class) %>% #Just keep datetime and prediction variables
  rename(type=.pred_class) #rename pred to count (for submission to Kaggle)


## Write out file
vroom_write(x=kaggle_submission, file="./GGGnn_2.csv", delim=",")




