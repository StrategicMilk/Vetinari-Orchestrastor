def process_data(input_path, output_path):
    # Read input, transform, write to output
    with open(input_path) as f_in:
        data = f_in.read()
    
    processed_data = data.upper()  # Example transformation

    with open(output_path, 'w') as f_out:
        f_out.write(processed_data)