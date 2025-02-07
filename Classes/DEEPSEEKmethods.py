import os
import pandas as pd
import openai
from openai import OpenAI
import json
import logging
import re
import time


class DEEPSEEKmethods:
    def __init__(self, params):
        """
        Initialize the class with the provided parameters.
        The constructor sets up the DEEPSEEK API key, model configuration, and various other
        parameters needed for generating prompts and making predictions.

        Args:
            params (dict): A dictionary containing the configuration settings.
        """
        # Access the OpenAI API key from environment variables
        # openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")

        # Initialize class variables using the provided parameters
        self.model_id = params['model_id']  # The model ID to use (e.g., deepseek-chat)
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
    Creates a training and validation JSONL file for DeepSeek fine-tuning.
    The method reads a CSV dataset, generates prompt-completion pairs for each row, and formats the data into
    the required JSONL structure for DeepSeek fine-tuning.
    The generated JSONL file will contain system, user, and assistant messages for each training || validation instance.
    """

    def create_jsonl(self, data_type, data_set):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.pre_path + data_set)
        data = []  # List to store the formatted data for each row

        # Iterate over each row in the DataFrame to format the data for fine-tuning
        for index, row in df.iterrows():
            data.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": self.system  # System message for context
                        },
                        {
                            "role": "user",
                            "content": self.generate_prompt(feature=row[self.feature_col])  # Generate user prompt
                        },
                        {
                            "role": "assistant",
                            "content": f"{{{self.json_key}: {row[self.label_col]}}}"  # Assistant's response
                        }
                    ]
                }
            )

        # Define the output file path for the JSONL file
        output_file_path = self.pre_path + "ft_dataset_deepseek_" + data_type + ".jsonl"  # Define the path
        # Write the formatted data to the JSONL file
        with open(output_file_path, 'w') as output_file:
            for record in data:
                # Convert each dictionary record to a JSON string and write it to the file
                json_record = json.dumps(record)
                output_file.write(json_record + '\n')

        # Return a success message with the file path
        return {"status": True, "data": f"JSONL file '{output_file_path}' has been created."}

    """
    Create a conversation with the DeepSeek model by sending a series of messages and receiving a response.
    This method constructs the conversation and returns the model's reply based on the provided messages.
    """

    def deepseek_conversation(self, conversation):
        try: # DeepSeek uses OpenAI's SDK
            client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

            completion = client.chat.completions.create(
                model=self.model_id,
                messages=conversation,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error getting response from DeepSeek: {e}")
            return None

    """
    Cleans the response from the DeepSeek model by attempting to extract and parse a JSON string.
    If the response is already in dictionary format, it is returned directly.
    If the response contains a JSON string, it will be extracted, cleaned, and parsed.
    If no valid JSON is found or a decoding error occurs, an error message is logged.
    """

    def clean_response(self, response, a_field):
        if isinstance(response, dict):  # If already valid, return it
            return {"status": True, "data": response}

        try:
            if response is None:
                raise ValueError("Response is None")

            start_index = response.find('{')
            end_index = response.rfind('}')

            if start_index != -1 and end_index != -1:
                json_str = response[start_index:end_index + 1]
                json_str = re.sub(r"\'", '"', json_str)  # Fix single quotes
                json_data = json.loads(json_str)  # Parse JSON
                return {"status": True, "data": json_data}
            else:
                logging.error(f"No JSON found in response. Input: '{a_field}', Response: {response}")
                return {"status": False, "data": f"No JSON found in response. Input: '{a_field}', Response: {response}"}

        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"JSON parsing error: {str(e)}. Input: '{a_field}', Response: {response}")
            return {"status": False, "data": f"JSON parsing error: {str(e)}. Input: '{a_field}', Response: {response}"}

    """
    Prompts the DeepSeek model to generate a prediction based on the provided input.
    The method constructs a conversation with the model using the system message and user input, 
    and processes the model's response to return a clean, formatted prediction.
    """

    def deepseek_prediction(self, input):
        conversation = [
            {'role': 'system', 'content': self.system},
            {'role': 'user', 'content': self.generate_prompt(feature=input[self.feature_col])}
        ]

        while True:
            conversation_response = self.deepseek_conversation(conversation)  # Call the model

            if conversation_response is None:
                logging.warning("Received None response. Retrying...")
                time.sleep(1)  # Short delay before retrying
                continue

            cleaned_response = self.clean_response(response=conversation_response, a_field=input[self.feature_col])

            if cleaned_response["status"]:  # If response is valid, return it
                return cleaned_response

            logging.warning("Invalid response format. Retrying...")
            time.sleep(1)  # Short delay before retrying

    """
    Makes predictions for a specific dataset and append the predictions to a new column.
    This method processes each row in the dataset, generates predictions using the DeepSeek model, 
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

                prediction = self.deepseek_prediction(input=row)

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


# TODO: Before running the script:
#  Ensure the DEEPSEEK_API_KEY is set as an environment variable to enable access to the DEEPSEEK API.

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
    'model_id': 'deepseek-chat',  # Specifies the DeepSeek model ID for making predictions.
    'prediction_column': 'deepseek-chat_prediction',  # Specifies the column where predictions will be stored.
    'pre_path': 'Datasets/',  # Specifies the base directory path where dataset files are located.
    'data_set': 'dataset_DEEP.csv',  # Defines the path to the CSV dataset file.
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

"""
Create an instance of the DEEPSEEKmethods class, passing the `params` dictionary to the constructor for initialization.
"""
DEEPSEEK = DEEPSEEKmethods(params)

"""
Call the `predictions` method of the DEEPSEEKmethods instance to make predictions on the specified dataset.
"""
DEEPSEEK.predictions()
