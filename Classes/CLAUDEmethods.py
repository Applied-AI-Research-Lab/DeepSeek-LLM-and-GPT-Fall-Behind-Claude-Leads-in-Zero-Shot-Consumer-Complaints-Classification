import anthropic
import json
import logging
import pandas as pd
import os
import re
import time

class CLAUDEmethods:
    def __init__(self, params):
        """
        Initialize the class with the provided parameters.
        The constructor sets up the model configuration, and various other
        parameters needed for generating prompts and making predictions.

        Args:
            params (dict): A dictionary containing the configuration settings.
        """

        # Initialize class variables using the provided parameters
        self.model_id = params['model_id']  # The model ID to use (e.g., claude-3-5-sonnet-20241022)
        self.prediction_column = params['prediction_column']  # Specifies the column where predictions will be stored
        self.pre_path = params['pre_path']  # The path to datasets
        self.data_set = params['data_set']  # Defines the path to the CSV dataset file
        self.prompt_array = params['prompt_array']  # A dictionary with additional data
        self.system = params['system']  # System-level message for context in the conversation
        self.prompt = params['prompt']  # The base prompt template
        self.feature_col = params['feature_col']  # Column name for feature input
        self.label_col = params['label_col']  # Column name for the label
        self.json_key = params['json_key']  # Key for extracting relevant data from the model's response
        self.max_tokens = params['max_tokens']  # Maximum number of tokens to generate in the response
        self.temperature = params['temperature']  # Controls response randomness (0 is most deterministic)

    """
    Generates a custom prompt
    """

    def generate_prompt(self, feature):
        # Create a new dictionary with the product title and existing categories
        replacement = '["retail_banking", "credit_reporting", "credit_card", "mortgages_and_loans", "debt_collection"]'

        updated_prompt = self.prompt.replace('[categories]', replacement)

        updated_prompt = updated_prompt + feature

        # If the prompt is simple you can avoid this method by setting updated_prompt = self.prompt + feature
        return updated_prompt  # This method returns the whole new custom prompt

    """
    Create a conversation with the Claude model by sending a series of messages and receiving a response.
    This method constructs the conversation and returns the model's reply based on the provided messages.
    """

    def claude_conversation(self, conversation):
        # Initialize the Anthropic API client
        client = anthropic.Anthropic()
        # Create a message request to send to the Claude model
        message = client.messages.create(
            model=self.model_id,  # The Claude model ID specified in the class instance
            max_tokens=self.max_tokens,  # Maximum number of tokens to generate in the response
            temperature=self.temperature,  # Control the variability of the response (higher means more creative)
            system=self.system,  # System-level context or instructions for the Claude model
            messages=conversation  # Pass the conversation history as input

            # or specify the type=text
            # messages=[ # The list of messages representing the conversation
            #     {
            #         "role": "user",
            #         "content": [
            #             {
            #                 "type": "text", # Specify the content type as text
            #                 "text": conversation # The actual conversation text being sent
            #             }
            #         ]
            #     }
            # ]
        )
        # Extract the `content` field from the API response
        data = message.content  # This is assumed to be a list of TextBlock objects
        # Extract only the textual part of each TextBlock object
        texts_only = [block.text for block in data]

        # Join the extracted text into a single string with line breaks (if there are multiple parts)
        text_only = "\n".join(texts_only)

        # Return the message from the model's response
        return text_only

    """
    Cleans the response from the Claude model by attempting to extract and parse a JSON string.
    If the response is already in dictionary format, it is returned directly.
    If the response contains a JSON string, it will be extracted, cleaned, and parsed.
    If no valid JSON is found or a decoding error occurs, an error message is logged.
    """

    def clean_response(self, response, a_field):
        # If the response is already a dictionary, return it directly
        if isinstance(response, dict):
            return {"status": True, "data": response}

        try:
            # Attempt to extract the JSON part from the response string
            start_index = response.find('{')
            end_index = response.rfind('}')

            if start_index != -1 and end_index != -1:
                # Extract and clean the JSON string
                json_str = response[start_index:end_index + 1]

                # Replace single quotes with double quotes
                json_str = re.sub(r"\'", '"', json_str)

                # Try parsing the cleaned JSON string
                json_data = json.loads(json_str)
                return {"status": True, "data": json_data}
            else:
                # Log an error if no JSON is found
                logging.error(f"No JSON found in the response. The input '{a_field}', resulted in the "
                              f"following response: {response}")
                return {"status": False, "data": f"No JSON found in the response. The input '{a_field}', "
                                                 f"resulted in the following response: {response}"}
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors
            logging.error(f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
                          f"resulted in the following response: {response}")
            return {"status": False,
                    "data": f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
                            f"resulted in the following response: {response}"}

    """
    Prompts the Claude model to generate a prediction based on the provided input.
    The method constructs a conversation with the model using the system message and user input, 
    and processes the model's response to return a clean, formatted prediction.
    """

    def claude_prediction(self, input):
        conversation = []
        # Add user input to the conversation, generating the appropriate prompt
        conversation.append({'role': 'user',
                             'content': self.generate_prompt(feature=input[self.feature_col])})  # Generate the prompt
        # Get the model's response by passing the conversation to claude_conversation
        conversation = self.claude_conversation(conversation)
        # Extract the content of the Claude model's response
        content = conversation

        # Clean and format the response before returning it
        return self.clean_response(response=content, a_field=input[self.feature_col])

    """
    Makes predictions for a specific dataset and append the predictions to a new column.
    This method processes each row in the dataset, generates predictions using the Claude model, 
    and updates the dataset with the predicted values in the specified prediction column.
    """

    def predictions(self):

        # Read the CSV dataset into a pandas DataFrame
        df = pd.read_csv(self.pre_path + self.data_set)

        # Create a copy of the original dataset (with '_original' appended to the filename)
        file_name_without_extension = os.path.splitext(os.path.basename(self.data_set))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original.csv'
        if not os.path.exists(original_file_path):
            os.rename(self.pre_path + self.data_set, original_file_path)

        # Check if the prediction_column is already present in the header
        if self.prediction_column not in df.columns:
            # If not, add the column to the DataFrame with pd.NA as the initial value
            df[self.prediction_column] = pd.NA

        if "time-" + self.model_id not in df.columns:
            df["time-" + self.model_id] = pd.NA

            # # Explicitly set the column type to a nullable integer
            # df = df.astype({prediction_column: 'Int64'})

        # Save the updated DataFrame back to CSV (if a new column is added)
        if self.prediction_column not in df.columns:
            df.to_csv(self.pre_path + self.data_set, index=False)

        # Set the dtype of the reason column to object
        # df = df.astype({reason_column: 'object'})

        # Iterate over each row in the DataFrame to make predictions
        for index, row in df.iterrows():
            # Make a prediction if the value in the prediction column is missing (NaN)
            if pd.isnull(row[self.prediction_column]):

                start_time = time.time()  # Start timer

                prediction = self.claude_prediction(input=row)

                end_time = time.time()  # End timer
                elapsed_time = round(end_time - start_time, 4)  # Compute elapsed time (rounded to 4 decimal places)

                # If the prediction fails, log the error and break the loop
                if not prediction['status']:
                    print(prediction)
                    break
                else:
                    print(prediction)
                    # If the prediction data contains a valid value, update the DataFrame
                    if prediction['data'][self.json_key] != '':
                        # Update the CSV file with the new prediction values
                        df.at[index, self.prediction_column] = prediction['data'][self.json_key]
                        # for integers only
                        # df.at[index, prediction_column] = int(prediction['data'][self.json_key])

                        df.at[index, "time-" + self.model_id] = elapsed_time

                        # Update the CSV file with the new values
                        df.to_csv(self.pre_path + self.data_set, index=False)
                    else:
                        logging.error(
                            f"No {self.json_key} instance was found within the data for '{row[self.feature_col]}', and the "
                            f"corresponding prediction response was: {prediction}.")
                        return {"status": False,
                                "data": f"No {self.json_key} instance was found within the data for '{row[self.feature_col]}', "
                                        f"and the corresponding prediction response was: {prediction}."}

                # break
            # Add a delay of 5 seconds (reduced for testing)

        # Change the column datatype after processing all predictions to handle 2.0 ratings
        # df[prediction_column] = df[prediction_column].astype('Int64')

        # After all predictions are made, return a success message
        return {"status": True, "data": 'Prediction have successfully been'}


# TODO! Before running the script:
#  1. Create a virtual environment:
#    python3 -m venv claude-env
#  2. Install the necessary packages:
#    pip install anthropic
#  3. Set the ANTHROPIC_API_KEY in the environment variables to enable API access.


"""
Configure the logging module to record error messages in a file named 'error_log.txt'.
"""
logging.basicConfig(filename='../error_log.txt', level=logging.ERROR)

"""
The `params` dictionary contains configuration settings for the AI model's prediction process. 
It includes specifications for the model ID, dataset details, system and task-specific prompts, 
and parameters for prediction output, response format, and model behavior.
"""
params = {
    # Specifies the Claude model ID for making predictions.
    # claude-3-5-sonnet-20241022 (claude-3-5-sonnet-latest) or claude-3-5-haiku-20241022 (claude-3-5-haiku-latest)
    'model_id': 'claude-3-5-sonnet-20241022',
    'prediction_column': 'claude_3.5_sonnet_prediction',  # Specifies the column where predictions will be stored.
    'pre_path': 'Datasets/',  # Specifies the base directory path where dataset files are located.
    'data_set': 'dataset.csv',  # Defines the path to the CSV dataset file.
    'prompt_array': {},  # Can be an empty array for simple projects.
    # Defines the system prompt that describes the task.
    'system': 'You are an AI assistant specializing in consumer complaint classification.',
    # Defines the prompt for the model, instructing it to make predictions and return its response in JSON format.
    # You can pass anything within brackets [example], which will be replaced during generate_prompt().
    'prompt': 'You are an AI assistant specializing in consumer complaint classification. Your task is to analyze a consumer complaint and classify it into the most appropriate category from the predefined list: [categories]. Provide your final classification in the following JSON format without explanations: {"product": "chosen_category_name"}. \nComplaint: ',
    'feature_col': 'narrative',  # Specifies the column in the dataset containing the text input/feature for predictions.
    'label_col': 'product',  # Used only for creating training and validation prompt-completion pairs JSONL files.
    'json_key': 'product',  # Defines the key in the JSON response expected from the model, e.g. {"category": "value"}
    'max_tokens': 1000,  # Sets the maximum number of tokens the model should generate in its response.
    'temperature': 0,  # Sets the temperature for response variability; 0 provides the most deterministic response.
}

# params['model_id'] = 'claude-3-5-sonnet-latest'
# params['prediction_column'] = 'claude_3_5_sonnet_prediction'

# params['model_id'] = 'claude-3-5-haiku-latest'
# params['prediction_column'] = 'claude_3_5_haiku_prediction' # claude_3_5_haiku_prediction

# params['model_id'] = 'claude-3-7-sonnet-latest'
# params['prediction_column'] = 'claude-3-7-sonnet_prediction'

"""
Create an instance of the CLAUDEmethods class, passing the `params` dictionary to the constructor for initialization.
"""
CLAUDE = CLAUDEmethods(params)

"""
Call the `predictions` method of the CLAUDEmethods instance to make predictions on the specified dataset.
"""
CLAUDE.predictions()
