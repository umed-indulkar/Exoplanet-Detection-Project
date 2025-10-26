# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # --- SETTINGS ---
# data_dir = "data/lightcurves_ultraclean_5000"   # or "processed_lightcurves"
# plot_count = 15   # how many curves to visualize

# # --- LOAD FILES ---
# files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
# print(f"Found {len(files)} cleaned lightcurves.")

# for i, file in enumerate(files[:plot_count]):
#     path = os.path.join(data_dir, file)
#     data = np.load(path)

#     time = data["time"]
#     flux = data["flux"]

#     plt.figure(figsize=(10,4))
#     plt.plot(time, flux, "r.", alpha=0.6, markersize=3)
#     plt.xlabel("Time / Phase [days]")
#     plt.ylabel("Normalized Flux")
#     plt.title(f"Ultra-clean lightcurve: {file}")
#     plt.grid(alpha=0.3)
#     plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt

# --- SETTINGS ---
data_dir = "lightcurve_ultraclean_5000"   # or "processed_lightcurves"
output_dir = "lightcurve_plots"  # where to save images

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# --- LOAD FILES ---
files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
print(f"\n{'='*60}")
print(f"Found {len(files)} cleaned lightcurves.")
print(f"{'='*60}\n")

# Interactive loop
current_idx = 0
while current_idx < len(files):
    file = files[current_idx]
    path = os.path.join(data_dir, file)
    data = np.load(path)
    
    time = data["time"]
    flux = data["flux"]
    
    # Extract curve ID from filename
    curve_id = file.replace(".npz", "")
    
    # Display info
    print(f"\n--- Curve {current_idx + 1}/{len(files)} ---")
    print(f"ID: {curve_id}")
    print(f"Data points: {len(time)}")
    print(f"Time range: {time.min():.2f} to {time.max():.2f} days")
    print(f"Flux range: {flux.min():.4f} to {flux.max():.4f}")
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(time, flux, "r.", alpha=0.6, markersize=3)
    plt.xlabel("Time / Phase [days]", fontsize=11)
    plt.ylabel("Normalized Flux", fontsize=11)
    plt.title(f"Lightcurve ID: {curve_id}", fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Show plot
    plt.show(block=False)
    plt.pause(0.1)
    
    # User prompt
    print("\n" + "="*60)
    print("COMMANDS:")
    print("="*60)
    print("  [n]  Next curve")
    print("  [p]  Previous curve")
    print("  [s]  Save current plot (without label)")
    print("  [l]  Add label and save")
    print("  [j]  Jump to specific curve number")
    print("  [f]  First curve (go to beginning)")
    print("  [e]  Last curve (go to end)")
    print("  [i]  Show detailed info about current curve")
    print("  [v]  View labels file")
    print("  [d]  Delete last saved label")
    print("  [r]  Refresh/redraw current plot")
    print("  [h]  Show help/commands")
    print("  [q]  Quit viewer")
    print("="*60)
    
    choice = input("\nYour choice: ").strip().lower()
    
    if choice == 'n':
        current_idx += 1
        if current_idx >= len(files):
            print("\n⚠ Reached end of dataset!")
            current_idx = len(files) - 1
            continue
    
    elif choice == 'p':
        current_idx -= 1
        if current_idx < 0:
            print("\n⚠ Already at first curve!")
            current_idx = 0
            continue
    
    elif choice == 'f':
        current_idx = 0
        print("\n↻ Jumped to first curve")
    
    elif choice == 'e':
        current_idx = len(files) - 1
        print("\n↻ Jumped to last curve")
    
    elif choice == 's':
        output_path = os.path.join(output_dir, f"{curve_id}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {output_path}")
        current_idx += 1
    
    elif choice == 'l':
        label = input("Enter label for this curve: ").strip()
        if label:
            # Add label to plot
            plt.suptitle(f"Label: {label}", fontsize=10, color='blue', y=0.98)
            output_path = os.path.join(output_dir, f"{curve_id}__{label}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved with label to: {output_path}")
            
            # Also save label to text file
            label_file = os.path.join(output_dir, "labels.txt")
            with open(label_file, 'a') as f:
                f.write(f"{curve_id}\t{label}\n")
            print(f"✓ Label appended to: {label_file}")
        else:
            print("⚠ No label entered, skipping save")
        current_idx += 1
    
    elif choice == 'j':
        try:
            jump_to = int(input(f"Jump to curve number (1-{len(files)}): ")) - 1
            if 0 <= jump_to < len(files):
                current_idx = jump_to
                print(f"↻ Jumped to curve {jump_to + 1}")
            else:
                print(f"⚠ Invalid number! Must be between 1 and {len(files)}")
                continue
        except ValueError:
            print("⚠ Invalid input!")
            continue
    
    elif choice == 'i':
        print("\n" + "="*60)
        print("DETAILED INFO:")
        print("="*60)
        print(f"Curve ID: {curve_id}")
        print(f"File: {file}")
        print(f"Position: {current_idx + 1} of {len(files)}")
        print(f"Data points: {len(time)}")
        print(f"Time range: {time.min():.4f} to {time.max():.4f} days")
        print(f"Time span: {time.max() - time.min():.4f} days")
        print(f"Flux range: {flux.min():.6f} to {flux.max():.6f}")
        print(f"Flux mean: {flux.mean():.6f}")
        print(f"Flux std: {flux.std():.6f}")
        print("="*60)
        input("\nPress Enter to continue...")
        continue
    
    elif choice == 'v':
        label_file = os.path.join(output_dir, "labels.txt")
        if os.path.exists(label_file):
            print("\n" + "="*60)
            print("SAVED LABELS:")
            print("="*60)
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    for line in lines:
                        print(line.strip())
                else:
                    print("No labels saved yet")
            print("="*60)
        else:
            print("\n⚠ No labels file found yet")
        input("\nPress Enter to continue...")
        continue
    
    elif choice == 'd':
        label_file = os.path.join(output_dir, "labels.txt")
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            if lines:
                removed = lines[-1].strip()
                with open(label_file, 'w') as f:
                    f.writelines(lines[:-1])
                print(f"✓ Removed last label: {removed}")
            else:
                print("⚠ No labels to delete")
        else:
            print("⚠ No labels file found")
        input("\nPress Enter to continue...")
        continue
    
    elif choice == 'r':
        print("↻ Refreshing plot...")
        continue
    
    elif choice == 'h':
        print("\n" + "="*60)
        print("HELP - COMMAND REFERENCE:")
        print("="*60)
        print("Navigation:")
        print("  n - Next curve (move forward)")
        print("  p - Previous curve (move backward)")
        print("  j - Jump to specific curve by number")
        print("  f - Jump to first curve")
        print("  e - Jump to last (end) curve")
        print("\nSaving:")
        print("  s - Save plot without label")
        print("  l - Save plot with custom label")
        print("\nInformation:")
        print("  i - Show detailed statistics for current curve")
        print("  v - View all saved labels")
        print("\nUtilities:")
        print("  d - Delete last saved label from labels.txt")
        print("  r - Refresh/redraw current plot")
        print("  h - Show this help menu")
        print("  q - Quit the viewer")
        print("="*60)
        input("\nPress Enter to continue...")
        continue
    
    elif choice == 'q':
        confirm = input("\nAre you sure you want to quit? (y/n): ").strip().lower()
        if confirm == 'y':
            print("\n👋 Exiting viewer. Goodbye!")
            break
        else:
            continue
    
    else:
        print("⚠ Invalid choice! Press 'h' for help")
        continue
    
    plt.close()

print(f"\n{'='*60}")
print("Session complete!")
print(f"{'='*60}\n")