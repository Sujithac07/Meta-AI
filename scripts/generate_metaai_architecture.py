import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_layer(ax, x, y, w, h, title, components, color):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="white",
        facecolor=color,
        alpha=0.95
    )
    ax.add_patch(box)
    text = f"{title}\n\n" + "\n".join([f"• {c}" for c in components])
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=11,
        fontweight="bold",
        linespacing=1.5,
    )


def add_arrow(ax, x, y_start, y_end):
    ax.annotate(
        "",
        xy=(x, y_end),
        xytext=(x, y_start),
        arrowprops=dict(arrowstyle="->", color="white", lw=2.0),
    )


def generate_architecture_diagram(output_path="docs/metaai_architecture.png"):
    fig, ax = plt.subplots(figsize=(12, 16))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.97,
        "MetaAI System Architecture",
        ha="center",
        va="center",
        color="white",
        fontsize=20,
        fontweight="bold",
    )

    layers = [
        ("Input Layer", ["CSV Upload", "API Request"], "#1d4ed8", 0.82),
        ("Agent Layer", ["Data Agent", "Model Agent", "Evaluation Agent", "Orchestrator"], "#7c3aed", 0.64),
        ("Core Layer", ["Meta-Learner", "Model Training", "Explainability Engine", "Drift Detector"], "#059669", 0.46),
        ("MLOps Layer", ["MLflow Tracking", "Model Registry", "Benchmark Runner"], "#d97706", 0.28),
        ("Output Layer", ["Gradio UI", "FastAPI", "PDF Report"], "#db2777", 0.10),
    ]

    x, w, h = 0.14, 0.72, 0.12
    for title, components, color, y in layers:
        draw_layer(ax, x, y, w, h, title, components, color)

    center_x = x + w / 2
    add_arrow(ax, center_x, 0.82, 0.76)
    add_arrow(ax, center_x, 0.64, 0.58)
    add_arrow(ax, center_x, 0.46, 0.40)
    add_arrow(ax, center_x, 0.28, 0.22)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved architecture diagram to: {output_path}")


if __name__ == "__main__":
    generate_architecture_diagram()

