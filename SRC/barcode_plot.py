import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def barcodeplot(data, file, file_name):
    colors = ['#270452', '#A82C2C', '#D49057', '#F1EDA6', '#FFFFFF']  # Ensure a fifth color (e.g., white for unused)
    category_labels = ['Sedentair', 'Licht intensief', 'Gemiddeld intensief', 'Hoog intensief', 'Niet gedragen']

    # Map fixed categories to colors (using an index for consistency)
    fixed_category_order = [0, 1, 2, 3, 4]  # Adjust based on unique categories in your dataset
    color_map = {category: colors[i] for i, category in enumerate(fixed_category_order)}

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 3))

    # Plot vertical lines for each activity category
    for i, category in enumerate(data['activities [1min]']):
        ax.vlines(data['time'][i], ymin=0, ymax=1, color=color_map.get(category, '#FFFFFF'), linewidth=5)

    # Set axis labels and title
    ax.set_title(f'Activiteiten voor bestand: {file}')
    ax.set_ylabel('Activiteiten')
    ax.set_xlabel('Tijd [HH:MM]')
    ax.set_yticks([])  # Remove y-axis ticks since this is categorical data

    # Format x-axis to show only hours and minutes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Remove unnecessary spines
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)

    # Create legend with fixed order and colors
    legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=4, label=category_labels[i]) for i in range(5)]
    ax.legend(handles=legend_elements, title='Categoriën', loc='upper right')

    # Show and save the plot
    plt.show()
    fig.savefig(f'Figures/{file_name}_1min.png')
