from ml_pipeline.processing import *
from ml_pipeline.training import train_and_evaluate_model
from projectpro import preserve, model_snapshot



# Read and Process Data
processed_data, x, y, cv_folds = read_and_process_data()
preserve("715840")


print("Data read and processed!")

# Modeling and Evaluation of the different multilabel approaches

# Choose approach name from the following:
# "independent"
# "classifier_chains"
# "native_extra_trees"
# "native_neural_net"
# "multilabel_to_multiclass"


scores, model = train_and_evaluate_model(x,y,cv_folds, approach_name="independent")
model_snapshot("715840")
print("End")


