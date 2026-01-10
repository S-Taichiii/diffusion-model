from evaluate.ev_line import evaluate_line
from evaluate.ev_circle_and_arc import evaluate_circle, evaluate_arc

# line
line_img1_dir = "D:/2024_Satsuka/github/DiffusionModel/data/line_224x224_val"
line_img2_dir = "D:/2024_Satsuka/github/DiffusionModel/generated_by_cond/2026_01_10_17_42/line"
evaluate_line(line_img1_dir, line_img2_dir)

# circle
circle_img1_dir = "D:/2024_Satsuka/github/DiffusionModel/data/circle_224x224_val"
circle_img2_dir = "D:/2024_Satsuka/github/DiffusionModel/generated_by_cond/2026_01_10_17_42/circle"
evaluate_circle(circle_img1_dir, circle_img2_dir)


# arc
arc_img1_dir = "D:/2024_Satsuka/github/DiffusionModel/data/arc_224x224_val"
arc_img2_dir = "D:/2024_Satsuka/github/DiffusionModel/generated_by_cond/2026_01_10_17_42/arc"
evaluate_arc(arc_img1_dir, arc_img2_dir)