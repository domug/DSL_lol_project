from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models.annotations import Title
from bokeh.plotting import figure, output_notebook, show, ColumnDataSource
from bokeh.models import HoverTool, Slider
import numpy as np


blue_total_gold          = [176, 204, 198, 234, 263, 193, 210, 166, 191, 173, 167, 167, 154, 189, 142, 252, 312, 313, 325, 305]
red_total_gold           = [162, 219, 187, 202, 220, 193, 172, 179, 203, 193, 172, 174, 142, 183, 155, 244, 335, 337, 320, 300]
blue_current_gold        = [256, 231, 249, 260, 218, 193, 152, 139, 175, 151, 164, 161, 111, 149, 104, 197, 302, 284, 230, 285]
red_current_gold         = [275, 237, 226, 254, 221, 195, 170, 183, 171, 165, 171, 155, 115, 150, 132, 233, 304, 326, 353, 312]

red_total_minion_kills   = [14, 98, 119, 172, 181, 125, 81, 116, 139, 99, 134, 107, 87, 126, 95, 179, 238, 206, 230, 200]
blue_total_minion_kills  = [5, 75, 109, 159, 158, 117, 98, 99, 133, 93, 114, 111, 72, 108, 64, 132, 202, 200, 221, 187]
red_jungle_minion_killed  = [0, 40, 57, 93, 93, 82, 64, 48, 81, 73, 96, 78, 55, 78, 84, 114, 188, 166, 136, 154]
blue_assist              = [49, 44, 42, 66, 89, 61, 59, 78, 86, 61, 82, 82, 79, 105, 68, 150, 206, 207, 211, 167]

red_ward_placed          = [63, 81, 76, 104, 127, 111, 88, 62, 109, 82, 105, 100, 89, 75, 63, 135, 195, 235, 206, 182]
blue_ward_placed         = [62, 59, 63, 92, 87, 77, 62, 77, 103, 74, 93, 91, 65, 96, 67, 136, 192, 186, 142, 149]
red_assist               = [63, 48, 64, 93, 74, 57, 54, 71, 107, 81, 92, 93, 68, 83, 71, 150, 217, 225, 200, 200]
blue_jungle_minion_killed = [0, 30, 67, 71, 71, 68, 57, 67, 78, 59, 74, 74, 60, 76, 67, 128, 176, 199, 183, 159]
blue_death               = [13, 19, 46, 67, 64, 44, 61, 74, 77, 59, 85, 81, 57, 81, 54, 133, 206, 200, 193, 159]

red_total_level          = [0, 15, 33, 53, 60, 47, 33, 44, 41, 47, 65, 52, 50, 59, 52, 90, 144, 152, 149, 129]
blue_total_level         = [0, 17, 31, 49, 43, 31, 33, 33, 52, 47, 57, 57, 50, 61, 53, 94, 135, 143, 133, 118]
blue_ward_kill           = [7, 19, 22, 33, 41, 34, 43, 43, 61, 51, 60, 54, 37, 60, 41, 83, 132, 134, 126, 128]
red_ward_kill            = [14, 10, 16, 25, 34, 30, 34, 38, 42, 52, 61, 45, 43, 56, 42, 129, 145, 167, 106, 118]

blue_kill                = [13, 6, 36, 49, 48, 39, 47, 57, 65, 69, 72, 61, 50, 73, 70, 130, 198, 195, 179, 150]
red_kill                 = [7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
red_death                = [4, 1, 10, 1, 1, 1, 15, 19, 18, 1, 1, 35, 28, 35, 22, 48, 1, 1, 1, 1]

time = ["1.5", "3", "4.5", "6", "7.5", "9", "10.5", "12", "13.5", "15", 
        "16.5", "18", "19.5", "21", "22.5", "24", "25.5", "27", "28.5", "30"]

names = ["blue_total_gold", "red_total_gold", "blue_current_gold", "red_current_gold", "red_total_minion_kills",
        "blue_total_minion_kills", "red_ward_placed", "blue_ward_placed", "blue_assist", "red_assist", "blue_death",
        "red_death", "blue_jungle_minion_killed", "red_jungle_minion_killed", "red_total_level", "blue_total_level",
        "blue_ward_kill", "red_ward_kill", "blue_kill", "red_kill"]

list1 = [blue_total_gold, red_total_gold, blue_current_gold, red_current_gold, red_total_minion_kills,
        blue_total_minion_kills, red_ward_placed, blue_ward_placed, blue_assist, red_assist, blue_death,
        red_death, blue_jungle_minion_killed, red_jungle_minion_killed, red_total_level, blue_total_level,
        blue_ward_kill, red_ward_kill, blue_kill, red_kill]

def get_element(list1, index):
    elements = []
    for i in list1:
        elements.append(i[index])
    return elements




## Visualization
factors = names.copy()      
x = get_element(list1, 0)

source = ColumnDataSource({
    "F_Score": x,
    "feature": names,
    "minute": [time[0]] * 20
})


# Create Hover Tooltip
hover_tool = HoverTool(tooltips=[
    ("F_Score", "@F_Score"),
    ("feature name", "@feature"),
    ("minute", "@minute")
])

# Define figure object
p = figure(tools=[hover_tool], y_range=factors, x_range=[-1,400],
           width=1000, height=700)

# Add Glyphs
p.segment(0, "feature", "F_Score", "feature", source=source, line_width=2, line_color="green")
p.circle( "F_Score", "feature", source=source, size=10, fill_color="orange", line_color="green", line_width=3)

# Set title
t = Title()
t.text = "{}분의 변수중요도".format(1.5)
p.title = t

# Define the callback function
def callback(attr, old, new):
    source.data = {
        "F_Score": get_element(list1, int(slider.value) - 1),
        "feature": names,
        "minute": [time[int(slider.value) - 1]] * 20
    }
    t.text = "{}분의 변수중요도".format(int(slider.value) * 1.5)

# Create Slider Widget
slider = Slider(start=1, end=20, step=1, value=0)
slider.on_change("value", callback)


layout = layout([p], [slider])


# Add the plot to the application
curdoc().add_root(layout)

