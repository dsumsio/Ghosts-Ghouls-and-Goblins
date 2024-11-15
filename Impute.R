library(tidyverse)
library(tidymodels)
library(vroom)

traindata = vroom('train.csv')
testdata = vroom('train.csv')
missingvals = vroom('trainWithMissingValues.csv')

traindata$color = factor(traindata$color)
traindata$type = factor(traindata$type)
testdata$color = factor(testdata$color)
missingvals$color = factor(missingvals$color)
missingvals$type = factor(missingvals$type)



recipe <- recipe(type ~ ., data=missingvals) %>% 
  step_impute_knn(bone_length, impute_with = imp_vars(has_soul, color), neighbors =7) %>%
  step_impute_knn(rotting_flesh, impute_with = imp_vars(has_soul, color, bone_length), neighbors =7) %>%
  step_impute_knn(hair_length, impute_with = imp_vars(has_soul, color, bone_length, rotting_flesh), neighbors =8)


prep <- prep(recipe)
imputedSet <- bake(prep, new_data = missingvals)

rmse_vec(traindata[is.na(missingvals)], imputedSet[is.na(missingvals)])
