from src.constants import LABELS_MAP
from unittest.mock import patch
from dashboard import pipeline_training, resume_inference


label_indices = {v: k for k, v in LABELS_MAP.items()}

# Mock the 'requests' module for inference testing


@patch('requests.post')
def test_resume_inference(mock_post_request):
    # Simulate a successful response for inference
    mock_post_request.return_value.text = '1.0'

    with patch('requests.post'):
        resume_inference()

# Mock the 'TrainingPipeline' class for pipeline training testing


@patch('src.training.train_pipeline.TrainingPipeline.train')
@patch('src.training.train_pipeline.TrainingPipeline.render_confusion_matrix')
@patch('src.training.train_pipeline.TrainingPipeline.get_model_performance')
@patch('streamlit.success')
@patch('streamlit.warning')
def test_pipeline_training(mock_train, mock_render_cm, mock_get_performance, mock_success, mock_warning):
    # Simulate a successful pipeline training
    mock_train.return_value = None
    mock_render_cm.return_value = None
    mock_get_performance.return_value = (0.8, 0.75)
    pipeline_training()
    # Assert that success and warning functions are called
    mock_success.assert_called()
    mock_warning.assert_called()


@patch('src.training.train_pipeline.TrainingPipeline.train')
@patch('src.training.train_pipeline.TrainingPipeline.render_confusion_matrix')
@patch('src.training.train_pipeline.TrainingPipeline.get_model_performance')
@patch('streamlit.success')
@patch('streamlit.warning')
def test_pipeline_training(mock_train, mock_render_cm, mock_get_performance, mock_success, mock_warning):
    # Simulate a successful pipeline training with the "Save" button checked
    mock_train.return_value = None
    mock_render_cm.return_value = None
    mock_get_performance.return_value = (0.8, 0.75)

    # Set a flag to simulate that the "Save" button is checked
    save_button_checked = True

    # Call the pipeline_training function with the "Save" button flag
    pipeline_training(serialize=save_button_checked, model_name="my_model")

    # Assert that success and warning functions are called
    mock_success.assert_called()
    if save_button_checked:
        # If the "Save" button is checked, the pipeline should be saved
        mock_success.assert_called_with("Pipeline saved!")
    else:
        # If the "Save" button is not checked, display a warning message
        mock_warning.assert_called_with(
            "Don't forget to save the pipeline if needed. You can repeat the training anytime 🙂."
        )


# Run the test
test_pipeline_training()
