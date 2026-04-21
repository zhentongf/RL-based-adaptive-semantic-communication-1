import matplotlib.pyplot as plt
from fuzzy_logic import decide_use_nn

def draw_table(ax, snr_norm, title):
    speed_levels = [0.0, 0.5, 1.0]
    distance_levels = [0.0, 0.5, 1.0]

    row_labels = ["slow", "medium", "fast"]
    col_labels = ["near", "medium", "far"]

    # Build table content
    table_data = []
    for speed in speed_levels:
        row = []
        for dist in distance_levels:
            use_nn = decide_use_nn(snr_norm, dist, speed)
            row.append("semantic" if use_nn else "direct")
        table_data.append(row)

    # Build full table content including labels (Shift indices right)
    full_data = []
    # Row 0: corner label + col_labels
    full_data.append(["speed\\distance"] + col_labels)
    # Rows 1-3: row_label + table_data
    for i in range(len(row_labels)):
        full_data.append([row_labels[i]] + table_data[i])

    # Create table
    table = ax.table(
        cellText=full_data,
        loc='center',
        cellLoc='center'
    )

    # Style adjustments
    table.scale(1.4, 2.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1.2)
        cell.set_fontsize(10)

        # Header styling (Row 0 or Col 0)
        if row == 0 or col == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')  # light gray for headers

        # Optional light shading (IEEE-friendly) for data cells
        if row > 0 and col > 0:
            text = cell.get_text().get_text()
            if text == "semantic":
                cell.set_facecolor('#f2f2f2')  # light gray
            else:
                cell.set_facecolor('#ffffff')  # white

    ax.set_title(title, fontsize=12, y=0.82, ha='center')
    ax.axis('off')


# === Create ONE figure with 3 tables ===
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

draw_table(axes[0], 0.0, "SNR = Low")
draw_table(axes[1], 0.5, "SNR = Medium")
draw_table(axes[2], 1.0, "SNR = High")

plt.tight_layout()
plt.savefig('./graphs/fuzzy_logic_graph_img.png', dpi=300, bbox_inches='tight')
plt.show()