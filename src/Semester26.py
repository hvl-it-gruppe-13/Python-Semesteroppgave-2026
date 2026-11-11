# importing modules and packages
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import matplotlib.image as mpimg
from models.LinearReg_model import LR_model
from models.Ridge_model import Ridge_model
from models.RidgeCV_model import RidgeCV_model

#Modeller

LR_bergen = LR_model("Bergen")
Ridge_bergen = Ridge_model("Bergen")
RidgeCV_bergen = RidgeCV_model("Bergen")

LR_oslo = LR_model("Oslo")
Ridge_oslo = Ridge_model("Oslo")
RidgeCV_oslo = RidgeCV_model("Oslo")

LR_stavanger = LR_model("Stavanger")
Ridge_stavanger = Ridge_model("Stavanger")
RidgeCV_stavanger = RidgeCV_model("Stavanger")


def draw_the_map():

    ax_map.cla()
    img = mpimg.imread(f"../img/Semester26_{current_map}.jpg")
    ax_map.imshow(img, extent=(0, 100, 0, 100), aspect='auto')

    fig.suptitle(f"Hus priser {current_map} (per m2)", fontsize=20, y=0.98, x=0.63)

    x,y = marked_point
    ax_map.scatter(x, y, marker='+', c='blue', s=200)
    print(f"marked_point: {marked_point}")
    if marked_point == (0, 0):
        return


    model = current_city_model
    m2list = [50, 100, 150, 200]

    m2_price = model.predict_price(
        x, y, m2, house_type, standard_type
    ) / m2

    price = f"{round(m2_price * 1000, -2):,.0f}"

    price_m2 = [
        model.predict_price(
            x, y, _m2, house_type, standard_type
        ) / _m2
        for _m2 in m2list
    ]



    ax_map.text(x + 3, y - 1, price, color='blue', fontsize=15)
    ax_history.cla()
    ax_history.bar(m2list, price_m2, color='blue', width=5)
    ax_history.legend(["pris per m2"])

    plt.draw()


def on_city_model(key):
    global current_city_model
    current_city_model = city_models[key]
    draw_the_map()

def on_map_type(key):
    global current_map, current_city_model
    current_map = key
    current_city_model = city_models[current_map][current_model]
    draw_the_map()

def on_model_type(key):
    global current_model, current_city_model
    current_model = key
    current_city_model = city_models[current_map][current_model]
    draw_the_map()

def on_click_map(event) :
    global marked_point
    if event.inaxes != ax_map:
        return
    marked_point = (event.xdata, event.ydata)
    draw_the_map()
    return

def on_house_type(key):
    global house_type
    house_type = house_types[key]
    draw_the_map()

def on_standard(key):
    global standard_type
    standard_type = standard_types[key]
    draw_the_map()

city_models = {
    "Bergen": {
        "LinearReg": LR_bergen,
        "Ridge": Ridge_bergen,
        "RidgeCV": RidgeCV_bergen
    },
    "Oslo": {
        "LinearReg": LR_oslo,
        "Ridge": Ridge_oslo,
        "RidgeCV": RidgeCV_oslo
    },
    "Stavanger": {
        "LinearReg": LR_stavanger,
        "Ridge": Ridge_stavanger,
        "RidgeCV": RidgeCV_stavanger
    }
}

maps = {
    "Bergen": "Bergen",
    "Stavanger": "Stavanger",
    "Oslo": "Oslo"
}

models = {
    "LinearReg": "LinearReg",
    "Ridge": "Ridge",
    "RidgeCV": "RidgeCV"
}

house_types = {
    'Enebolig': 1,
    'Rekke': 2,
    'Leilighet': 3
}
standard_types = {
    'Dårlig': 1,
    'God': 2,
    'Meget God': 3,
    'Luksus': 4
}

house_type = 1
standard_type = 1
m2 = 100
marked_point = (43,57)

current_map = "Bergen"
current_model = "LinearReg"

current_city_model = city_models[current_map][current_model]

# Create the figures
fig = plt.figure(figsize=(8.0, 8.0), facecolor='white')  # type: ignore[arg-type]

ax_map = fig.add_axes((0.40, 0.05, 0.55, 0.85))
ax_radio_map = fig.add_axes((0.05, 0.88, 0.30, 0.08))
ax_radio_house_type = fig.add_axes((0.05, 0.72, 0.30, 0.13))
ax_radio_standard = fig.add_axes((0.05, 0.56, 0.30, 0.13))
ax_radio_model = fig.add_axes((0.05, 0.40, 0.30, 0.13))
ax_history = fig.add_axes((0.05, 0.05, 0.30, 0.30))
ax_radio_house_type.set_facecolor('lightgray')
ax_radio_standard.set_facecolor('lightgray')
ax_radio_map.set_facecolor('lightgray')
ax_radio_model.set_facecolor('lightgray')

radio_button_category = RadioButtons(ax_radio_house_type, list(house_types.keys()))
radio_button_standard = RadioButtons(ax_radio_standard, list(standard_types.keys()))
radio_button_map = RadioButtons(ax_radio_map, list(maps.keys()))
radio_button_model = RadioButtons(ax_radio_model, list(models.keys()))
radio_button_category.on_clicked(on_house_type)
radio_button_standard.on_clicked(on_standard)
radio_button_map.on_clicked(on_map_type)
radio_button_model.on_clicked(on_model_type)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust the figure to fit the image
ax_map.margins(x=0.01, y=0.01)  # Adjust x and y margins



draw_the_map()
plt.connect('button_press_event', on_click_map)
plt.show()


