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

recipe <- recipe(type ~ ., data=train_data) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1) %>%
  step_smote(all_outcomes(), neighbors = 2)

nb_model <- naive_Bayes(Laplace= tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(nb_model)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(Laplace(), smoothness(),
                                      levels = 15)

## Split data for CV
folds <- vfold_cv(train_data, v = 5, repeats = 1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(roc_auc)
  )


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <- nb_wf %>%
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
vroom_write(x=kaggle_submission, file="./GGGnb_3.csv", delim=",")



