import joblib

# Load the model
model = joblib.load('model.joblib')
print("The model is :",model)
# If the model is a Pipeline and has preprocessing steps like ColumnTransformer
if hasattr(model, 'named_steps'):
    # If using ColumnTransformer or similar preprocessing step
    column_transformer = model.named_steps['preprocessor']  # assuming 'preprocessor' is the name of your transformer

    # If using ColumnTransformer, get the feature names
    try:
        column_names = column_transformer.get_feature_names_out()
    except AttributeError:
        # Handle older versions of scikit-learn
        column_names = column_transformer.get_feature_names()

    print(column_names)

# If there's no pipeline, but you saved the column names separately, load those
