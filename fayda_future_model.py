# Python-based implementation of future-state process time estimation model using LLM

# Step 1: Import the Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai
import os

# Step 2: Set up OpenAI API key
# Note: Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = 'your-api-key'

# Step 3: Define the function to load data and process future-state estimation

def estimate_processing_time(activity_name, context):
    """
    Function to use OpenAI LLM to estimate processing time for a given activity.
    """
    print(f"Estimating processing time for activity: {activity_name}")
    messages = [
        {"role": "system", "content": """You are an expert in process optimization for telecommunications services in Ethiopia. 
        Your task is to estimate realistic processing times for new activities in seconds.
        
        You MUST follow this exact format for your response:
        [number in seconds]
        [explanation]
        
        Example response:
        20
        This activity would take approximately 20 seconds because...
        """},
        
        {"role": "user", "content": f"""
        Context: The following are the current activities and their durations in an Ethiopian telecom center:
        {context}

        Task: Estimate how many seconds the following new activity would take:
        "{activity_name}"

        Consider:
        1. Similar activities in the original process above
        2. Technical infrastructure limitations in Ethiopia
        3. Staff training needs
        4. Customer interaction time
        5. Potential delays or retries

        Remember to respond ONLY in this format:
        [number in seconds]
        [explanation]
        """}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            max_tokens=250
        )
        
        response_text = response.choices[0].message['content'].strip()
        print(f"\nFull response for '{activity_name}':\n{response_text}")
        
        # Split response into lines and get the first non-empty line
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        if not lines:
            raise ValueError("Empty response received")
            
        first_line = lines[0]
        explanation = "\n".join(lines[1:])  # Join the rest as explanation
        
        # Try to convert the first line to a number (int or float)
        try:
            # Remove any non-numeric characters except decimal points
            numeric_str = ''.join(c for c in first_line if c.isdigit() or c == '.')
            estimated_time = float(numeric_str)  # Convert to float at the end for consistency
            print(f"Successfully extracted time estimate: {estimated_time} seconds")
            if estimated_time <= 0:
                raise ValueError("Time estimate must be positive")
            return estimated_time, explanation
            
        except ValueError as ve:
            print(f"Warning: Could not parse first line as number: '{first_line}'")
            # Fallback: try to find first number in the entire response
            import re
            numbers = re.findall(r'\b\d+\.?\d*\b', response_text)
            if numbers:
                estimated_time = float(numbers[0])
                if estimated_time > 0:
                    print(f"Fallback - extracted first number: {estimated_time} seconds")
                    return estimated_time, explanation
            return np.nan, explanation
            
    except Exception as e:
        print(f"Error estimating time for '{activity_name}': {str(e)}")
        return np.nan, ""

def find_processing_time(row, original_df, context):
    activity_name = row['Activity Name']
    
    # For new activities, use LLM estimation
    if row['Activity Status'] == 'New':
        estimated_time, explanation = estimate_processing_time(activity_name, context)
        row['Explanation'] = explanation
        return estimated_time
    
    # For existing activities, find match in original_df
    matching_activities = original_df.loc[original_df['Activity Name'] == activity_name]
    
    if len(matching_activities) == 0:
        print(f"\nWARNING: No matching activity found in original data for: '{activity_name}'")
        print("Available activities in original data:")
        for act in original_df['Activity Name'].values:
            print(f"- {act}")
        return np.nan  # or some default value like 0
    
    return matching_activities['Processing Time (seconds)'].iloc[0]

# Step 4: Loop through multiple input files for Original and Future-State Data
input_folder = os.path.join('input')
output_folder = os.path.join('output')

# Get all files from the input folder
print("Fetching input files...")
original_files = sorted([f for f in os.listdir(input_folder) if f.startswith('original_data_') and f.endswith('.xlsx')])
future_files = sorted([f for f in os.listdir(input_folder) if f.startswith('future_data_') and f.endswith('.xlsx')])

print(f"Found {len(original_files)} original data files and {len(future_files)} future data files.")

# Ensure the number of original and future files match
if len(original_files) != len(future_files):
    raise ValueError("The number of original and future data files do not match.")

# Iterate over each pair of original and future files
for i, (original_file, future_file) in enumerate(zip(original_files, future_files)):
    print(f"\nProcessing file pair {i+1}: Original File - {original_file}, Future File - {future_file}")
    # Load data into DataFrames
    original_data_path = os.path.join(input_folder, original_file)
    future_data_path = os.path.join(input_folder, future_file)
    
    print(f"Loading original data from {original_data_path}")
    original_df = pd.read_excel(original_data_path)
    print(f"Loading future data from {future_data_path}")
    future_df = pd.read_excel(future_data_path)

    # Prepare the context string for LLM (original activities and times)
    context = ", ".join([f"{row['Activity Name']} ({row['Processing Time (seconds)']} seconds)" for index, row in original_df.iterrows()])
    print(f"Context for LLM: {context}")

    # After loading the DataFrames
    print("\nOriginal Activities:")
    for act in original_df['Activity Name'].values:
        print(f"- {act}")

    print("\nFuture Activities:")
    for _, row in future_df.iterrows():
        print(f"- {row['Activity Name']} (Status: {row['Activity Status']})")

    # Add a new column for explanations
    future_df['Explanation'] = ""

    # Estimate times for future-state activities using LLM
    print("Estimating processing times for future-state activities...")
    future_df['Processing Time (seconds)'] = future_df.apply(
        lambda row: find_processing_time(row, original_df, context),
        axis=1
    )

    # After processing, check for any NaN values
    nan_activities = future_df[future_df['Processing Time (seconds)'].isna()]
    if not nan_activities.empty:
        print("\nWARNING: The following activities could not be processed:")
        for _, row in nan_activities.iterrows():
            print(f"- {row['Activity Name']} (Status: {row['Activity Status']})")

    # Step 5: Analyze the Future-State Model
    # Calculate Total Processing Time for Original and Future-State Processes
    original_total_time = original_df['Processing Time (seconds)'].sum()
    future_total_time = future_df['Processing Time (seconds)'].sum()

    print(f"Total Processing Time for Original Process (File {i+1}): {original_total_time} seconds")
    print(f"Total Processing Time for Future-State Process (LLM Estimated, File {i+1}): {future_total_time} seconds")

    # Extract tag from original file name
    tag = original_file.split('original_data_')[1].split('.xlsx')[0]

    # Step 6: Save the Future-State Estimates to Excel File
    # File path for output
    output_path = os.path.join(output_folder, f'future_state_estimates_{tag}.xlsx')
    print(f"Saving future-state estimates to {output_path}")
    future_df.to_excel(output_path, index=False)

    # Step 7: Visualization of Time Savings
    # Plotting a bar chart to compare original and future-state processing times
    activities = ['Original Process', f'Future-State Process (LLM Estimated, File {i+1})']
    times = [original_total_time, future_total_time]

    print(f"Visualizing time comparison for file pair {i+1}...")
    plt.figure(figsize=(10, 6))
    plt.bar(activities, times, color=['blue', 'green'])
    plt.xlabel('Process Type')
    plt.ylabel('Total Processing Time (seconds)')
    plt.title(f'Comparison of Original and Future-State Processing Times (LLM Estimated, File {i+1})')
    plt.show()

# Note: You will need to have access to the OpenAI API to use this implementation. Replace 'your-api-key' with your actual API key.
# The LLM estimates processing times for new activities based on reasoning through the original context and current technological trends.

# Step 8: Create Fake Excel Files for Demonstration Purposes
def create_fake_excel_files():
    # Define the structure of the original and future state data
    original_data = {
        'Activity Name': ['Fill Application Form', 'Verify Identity', 'Manual Record Keeping'],
        'Processing Time (seconds)': [120, 60, 90]
    }
    future_data = {
        'Activity Name': ['Fill Application Form', 'Verify Identity', 'ID Check Automation', 'Auto-Form Validation'],
        'Activity Status': ['Retained', 'Retained', 'New', 'New']
    }
    
    # Create DataFrames
    original_df = pd.DataFrame(original_data)
    future_df = pd.DataFrame(future_data)
    
    # Create a directory for the input files if it does not exist
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    
    # Save the DataFrames as Excel files
    original_df.to_excel(os.path.join(input_folder, 'original_data_1.xlsx'), index=False)
    future_df.to_excel(os.path.join(input_folder, 'future_data_1.xlsx'), index=False)
    print("Fake Excel files created in 'root/input' directory.")

# Uncomment the following line to create the fake excel files for demonstration purposes
# create_fake_excel_files()
