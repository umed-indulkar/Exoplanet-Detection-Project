import pandas as pd
import matplotlib.pyplot as plt

def plot_single_row(csv_row_string):
    # Convert your string output back to numbers
    data = [float(x) for x in csv_row_string.split(',')]
    id = data[0]  # First value is the ID
    label = data[1]  # Second value is the label
    flux = data[2:]  # Remaining values are the flux data

    plt.style.use('dark_background') # Professional "Space" look
    plt.figure(figsize=(10, 5))
    
    # Plot the binned points
    plt.plot(range(len(flux)), flux, color='#00ffcc', linewidth=2, marker='o', markersize=4)
    
    # Highlight the transit zone (Bins 225 to 275)
    plt.axvspan(225, 275, color='yellow', alpha=0.1, label="Transit Window")

    plt.suptitle(f"Light Curve ID: {int(id)}", color='white', size=16)
    plt.title(f"Processed Light Curve | Label: {int(label)}", color='white', size=14)
    plt.xlabel("Phase Bin", color='white')
    plt.ylabel("Normalized Intensity", color='white')
    plt.ylim(min(flux) - 0.001, max(flux) + 0.001)
    plt.grid(color='gray', linestyle='--', alpha=0.3)
    plt.legend()
    plt.show()

# Example: Paste one line from your CSV here to test
example_row = "" 
plot_single_row(example_row)