import pandas as pd
import ast
import re

def parse_log_file(log_file_path):
    """
    Parse the log file and create a pandas DataFrame with specified columns.
    
    Args:
        log_file_path (str): Path to the log file
    
    Returns:
        pd.DataFrame: DataFrame with columns: bisim_weight, epoch, success_rate, steps
    """
    data = []
    
    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    # Parse the dictionary-like string
                    log_entry = ast.literal_eval(line)
                    
                    # Extract relevant fields and map to desired column names
                    row = {
                        'bisim_weight': log_entry.get('bisim_weight'),
                        'epoch': log_entry.get('model_epoch'),
                        'success_rate': log_entry.get('final_success_rate'),
                        'steps': log_entry.get('total_steps')
                    }
                    data.append(row)
                    
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse line: {line}")
                    print(f"Error: {e}")
                    continue
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def display_table(df):
    """
    Display the table in a formatted way.
    
    Args:
        df (pd.DataFrame): The DataFrame to display
    """
    print("Parsed Log Data:")
    print("=" * 50)
    print(df.to_string(index=False))
    print(f"\nTotal entries: {len(df)}")

# Example usage
if __name__ == "__main__":
    # Replace with your actual log file path
    log_file_path = 'noChangeSeed42.log'
    
    try:
        # Parse the log file
        df = parse_log_file(log_file_path)
        
        # Display the table
        display_table(df)
        
        # Optional: Save to CSV
        df.to_csv('parsed_log_data.csv', index=False)
        print(f"\nData saved to 'parsed_log_data.csv'")
        
        # Optional: Display basic statistics
        print("\nBasic Statistics:")
        print("-" * 30)
        print(df.describe())
        
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found.")
        print("Please make sure the file path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")
