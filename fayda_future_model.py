# Python-based implementation of future-state process time estimation model using LLM

# Step 1: Import the Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai
import os
from collections import defaultdict
from activity_dependencies import ActivityDependencyAnalyzer
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def setup_logging():
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure log file path with timestamp
    log_file = os.path.join(log_dir, 'fayda_model.log')
    
    # Clear existing log file
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    
    logging.info('='*80)
    logging.info(f'Starting new run at {datetime.now()}')
    logging.info('='*80)

def estimate_processing_time(activity_name, context, service_provider, sector, business_process):
    """Function to use OpenAI LLM to estimate processing time for a given activity."""
    print(f"Estimating processing time for activity: {activity_name}")
    messages = [
        {"role": "system", "content": """You are an expert in process optimization for telecommunications, legal and banking services in Ethiopia. 
        Your task is to estimate realistic processing times for new activities in seconds.
        
        You MUST follow this exact format for your response:
        [number in seconds]
        [explanation]
        """},
        {"role": "user", "content": f"""
        Context: 
        Service Provider: {service_provider}
        Sector: {sector}
        Business Process: {business_process}
        
        The following are the current activities and their durations in this Ethiopian business process:
        {context}

        Task: Estimate how many seconds the following new activity would take:
        "{activity_name}"
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
        explanation = "\n".join(lines[1:])
        
        # Try to convert the first line to a number (int or float)
        try:
            # Remove any non-numeric characters except decimal points
            numeric_str = ''.join(c for c in first_line if c.isdigit() or c == '.')
            estimated_time = float(numeric_str)
            if estimated_time <= 0:
                raise ValueError("Time estimate must be positive")
            return estimated_time, explanation
            
        except ValueError as ve:
            print(f"Warning: Could not parse first line as number: '{first_line}'")
            import re
            numbers = re.findall(r'\b\d+\.?\d*\b', response_text)
            if numbers:
                estimated_time = float(numbers[0])
                if estimated_time > 0:
                    return estimated_time, explanation
            return np.nan, explanation
            
    except Exception as e:
        print(f"Error estimating time for '{activity_name}': {str(e)}")
        return np.nan, ""

def find_processing_time(row, original_df, context):
    """Find processing time for an existing activity"""
    activity_name = row['Activity Name']
    service_provider = row['Service Provider']
    sector = row['Sector']
    business_process = row['Business Process']
    
    matching_activities = original_df[
        (original_df['Activity Name'] == activity_name) &
        (original_df['Service Provider'] == service_provider) &
        (original_df['Sector'] == sector) &
        (original_df['Business Process'] == business_process)
    ]
    
    if len(matching_activities) == 0:
        logging.warning(f"\nWARNING: No matching activity found for: '{activity_name}' in {service_provider}, {sector}, {business_process}")
        return np.nan
    
    return matching_activities['Processing Time (seconds)'].iloc[0]

def process_group_with_dependencies(group, original_df, sp, sector, bp):
    """Process activities using dependency analysis"""
    analyzer = ActivityDependencyAnalyzer()
    
    # Get optimal processing order
    processing_order = analyzer.get_optimal_processing_order(group)
    logging.info(f"Processing {len(processing_order)} activities in optimized order")
    
    results = []
    context_activities = []
    
    # Add original activities to context
    original_context = original_df[
        (original_df['Service Provider'] == sp) &
        (original_df['Sector'] == sector) &
        (original_df['Business Process'] == bp)
    ]
    
    for _, row in original_context.iterrows():
        context_activities.append(
            f"{row['Activity Name']} ({row['Processing Time (seconds)']} seconds)"
        )
    
    # Process activities in dependency-aware order
    for activity_name in processing_order:
        row = group[group['Activity Name'] == activity_name].iloc[0]
        idx = row.name
        
        logging.info(f"\nProcessing activity: {activity_name}")
        
        if row['Activity Status'] == 'New':
            logging.debug(f"Estimating time for new activity: {activity_name}")
            time, explanation = estimate_processing_time(
                activity_name,
                ", ".join(context_activities),
                sp, sector, bp
            )
        else:
            logging.debug(f"Finding time for existing activity: {activity_name}")
            time = find_processing_time(
                row, 
                original_df,
                ", ".join(context_activities)
            )
            explanation = ""
        
        results.append((idx, time, explanation))
        if not pd.isna(time):
            context_activities.append(f"{activity_name} ({time} seconds)")
            logging.debug(f"Added {activity_name} to context. Context now has {len(context_activities)} activities")
    
    return results

def main():
    # Add at the start of main()
    setup_logging()
    
    # Set up paths
    input_folder = os.path.join('input')
    output_folder = os.path.join('output')
    
    logging.info(f"Input folder: {input_folder}")
    logging.info(f"Output folder: {output_folder}")
    
    # Load data
    original_data_path = os.path.join(input_folder, 'original_data_all.xlsx')
    future_data_path = os.path.join(input_folder, 'future_data_all.xlsx')
    
    logging.info(f"Loading original data from {original_data_path}")
    original_df = pd.read_excel(original_data_path)
    logging.info(f"Loaded {len(original_df)} original activities")
    
    logging.info(f"Loading future data from {future_data_path}")
    future_df = pd.read_excel(future_data_path)
    logging.info(f"Loaded {len(future_df)} future activities")
    
    # Initialize results columns
    future_df['Explanation'] = ""
    future_df['Processing Time (seconds)'] = np.nan
    
    # Process by group
    grouped_future = future_df.groupby(['Service Provider', 'Sector', 'Business Process'])
    logging.info(f"Processing {len(grouped_future)} groups")
    
    for (sp, sector, bp), group in grouped_future:
        logging.info(f"\nProcessing group: {sp} - {sector} - {bp}")
        
        results = process_group_with_dependencies(group, original_df, sp, sector, bp)
        
        # Update the dataframe with results
        for idx, time, explanation in results:
            future_df.loc[idx, 'Processing Time (seconds)'] = time
            future_df.loc[idx, 'Explanation'] = explanation
    
    # Calculate results
    original_total_time = original_df['Processing Time (seconds)'].sum()
    future_total_time = future_df['Processing Time (seconds)'].sum()
    
    logging.info("\nResults Summary:")
    logging.info(f"Total Processing Time for Original Process: {original_total_time} seconds")
    logging.info(f"Total Processing Time for Future-State Process: {future_total_time} seconds")
    logging.info(f"Time Difference: {original_total_time - future_total_time} seconds")
    
    # Save results
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, 'future_state_estimates.xlsx')
    future_df.to_excel(output_path, index=False)
    logging.info(f"\nSaved results to {output_path}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.bar(['Original Process', 'Future-State Process'], 
            [original_total_time, future_total_time],
            color=['blue', 'green'])
    plt.title('Comparison of Processing Times')
    plt.ylabel('Total Processing Time (seconds)')
    plt.show()
    
    logging.info("Analysis complete")

if __name__ == "__main__":
    main()