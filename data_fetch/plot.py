import pandas as pd
import matplotlib.pyplot as plt

def plot_single_row(csv_row_string):
    # Convert your string output back to numbers
    data = [float(x) for x in csv_row_string.split(',')]
    label = data[0]
    flux = data[1:]

    plt.style.use('dark_background') # Professional "Space" look
    plt.figure(figsize=(10, 5))
    
    # Plot the binned points
    plt.plot(range(len(flux)), flux, color='#00ffcc', linewidth=2, marker='o', markersize=4)
    
    # Highlight the transit zone (Bins 30 to 50)
    plt.axvspan(30, 50, color='yellow', alpha=0.1, label="Transit Window")
    
    plt.title(f"Processed Light Curve | Label: {int(label)}", color='white', size=14)
    plt.xlabel("Phase Bin", color='white')
    plt.ylabel("Normalized Intensity", color='white')
    plt.ylim(min(flux) - 0.001, max(flux) + 0.001)
    plt.grid(color='gray', linestyle='--', alpha=0.3)
    plt.legend()
    plt.show()

# Example: Paste one line from your CSV here to test
example_row = "0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9997534,0.999961,0.9998091,0.9999296,0.99988145,1.0001743,1.0002369,1.0002598,1.0002432,1.0003221,1.0004511,1.0003164,1.0003375,1.0000883,1.0001622,0.9998088,0.9998224,0.9997687,0.99965304,0.9996414,0.9998159,0.99981403,0.9999393,1.0001003,1.0000613,0.9999875,1.0001155,1.0000824,1.0000508,0.9998844,1.0000389,1.0000818,1.0000606,1.0000461,1.0000699,1.0000776,1.0000952,1.0001501,0.99991095,0.99982405,0.99981636,0.9999348,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0" 
plot_single_row(example_row)