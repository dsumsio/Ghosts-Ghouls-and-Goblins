library(bonsai)
library(lightgbm)
library(tidyverse)
library(tidymodels)
library(vroom)

sample = vroom('sample_submission.csv')
train_data = vroom('train.csv')
test_data = vroom('test.csv')
test_data$color = as.factor(test_data$color)


recipe <- recipe(type ~ ., data=train_data) %>%
  update_role(id, new_role="id") %>%
  step_mutate_at(color, fn = factor) %>% 
  step_dummy(color) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_zv(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_numeric_predictors(), threshold=0.8) # %>%
  # step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]


boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")




tuneGrid <- grid_regular(trees(),
                         tree_depth(),
                         learn_rate(),
                            levels=5)

folds <- vfold_cv(train_data, v = 5, repeats = 1)

wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(boost_model)

CV_results <- wf %>%
  tune_grid(
    resamples = folds,
    grid = tuneGrid,
    metrics = metric_set(roc_auc)
  )

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <- wf %>%
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
vroom_write(x=kaggle_submission, file="./GGGboost_3.csv", delim=",")

