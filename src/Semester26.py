
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RadioButtons
from matplotlib.ticker import MaxNLocator, FuncFormatter
from pyproj import Transformer

from models.LinearReg_model import LRModel
from models.Ridge_model import RidgeModel
from models.RidgeCV_model import RidgeCVModel


# Map images are expected in a "img" folder beside this file.
MAPPE = Path(__file__).resolve().parent.parent / "img"

# Map crop: left, top, right, bottom, measured in original image pixels.
BYER = {
    "Bergen": {
        "fil": "Semester26_Bergen.jpg",
        "senter": (-29533.35, 6731520.12),
        "kartflate": (11, 7, 873, 1198),
        "fire_km_px": 130,
        "maalestokk_km": 4,
    },
    "Oslo": {
        "fil": "Semester26_Oslo.jpg",
        "senter": (264646.26, 6649304.95),
        "kartflate": (33, 33, 989, 1343),
        "fire_km_px": 182,
        "maalestokk_km": 4,
    },
    "Stavanger": {
        "fil": "Semester26_Stavanger.jpg",
        "senter": (-35616.22, 6568552.35),
        "kartflate": (33, 33, 989, 1343),
        "fire_km_px": 182,
        "maalestokk_km": 4,
    },
}

TIL_WGS84 = Transformer.from_crs(
    "EPSG:25833",
    "EPSG:4326",
    always_xy=True,
)


# Create the three prediction models for each city.
LR_bergen = LRModel("Bergen")
Ridge_bergen = RidgeModel("Bergen")
RidgeCV_bergen = RidgeCVModel("Bergen")

LR_oslo = LRModel("Oslo")
Ridge_oslo = RidgeModel("Oslo")
RidgeCV_oslo = RidgeCVModel("Oslo")

LR_stavanger = LRModel("Stavanger")
Ridge_stavanger = RidgeModel("Stavanger")
RidgeCV_stavanger = RidgeCVModel("Stavanger")

city_models = {
    "Bergen": {
        "LinearReg": LR_bergen,
        "Ridge": Ridge_bergen,
        "RidgeCV": RidgeCV_bergen,
    },
    "Oslo": {
        "LinearReg": LR_oslo,
        "Ridge": Ridge_oslo,
        "RidgeCV": RidgeCV_oslo,
    },
    "Stavanger": {
        "LinearReg": LR_stavanger,
        "Ridge": Ridge_stavanger,
        "RidgeCV": RidgeCV_stavanger,
    },
}

house_types = {
    "Enebolig": 1,
    "Rekke": 2,
    "Leilighet": 3,
}

standard_types = {
    "Dårlig": 1,
    "God": 2,
    "Meget God": 3,
    "Luksus": 4,
}

model_names = ["LinearReg", "Ridge", "RidgeCV"]


# Current selections.
valgt_by = "Bergen"
valgt_modell = "LinearReg"
house_type = 1
standard_type = 1
m2 = 100

# This is in the original models' 0–100 map-coordinate scale.
# It preserves the first program's initial marked point.
marked_model_point = (43.0, 57.0)


def kartdata(by):
    """Load and crop a city map; return it with projected-coordinate bounds."""
    info = BYER[by]
    fil = MAPPE / info["fil"]

    if not fil.is_file():
        raise FileNotFoundError(
            f"Fant ikke {fil}. Kartbildene må ligge i mappen 'img'."
        )

    bilde = mpimg.imread(fil)
    x0, y0, x1, y1 = info["kartflate"]
    utsnitt = bilde[y0:y1, x0:x1]

    px_per_meter = info["fire_km_px"] / (info["maalestokk_km"] * 1000)
    bredde_meter = utsnitt.shape[1] / px_per_meter
    hoyde_meter = utsnitt.shape[0] / px_per_meter

    ost, nord = info["senter"]
    grenser = (
        ost - bredde_meter / 2,
        ost + bredde_meter / 2,
        nord - hoyde_meter / 2,
        nord + hoyde_meter / 2,
    )

    return utsnitt, grenser


def modellkoordinater(x, y, grenser):
    """Convert projected map coordinates to the models' 0–100 scale."""
    venstre, hoyre, bunn, topp = grenser
    model_x = (x - venstre) / (hoyre - venstre) * 100
    model_y = (y - bunn) / (topp - bunn) * 100
    return model_x, model_y


def lengdegrad(verdi, _):
    ost, nord = BYER[valgt_by]["senter"]
    lon, _lat = TIL_WGS84.transform(verdi, nord)
    return f"{lon:.3f}°"


def breddegrad(verdi, _):
    ost, nord = BYER[valgt_by]["senter"]
    _lon, lat = TIL_WGS84.transform(ost, verdi)
    return f"{lat:.3f}°"


def draw_the_map():
    ax_map.clear()

    bilde, grenser = kartdata(valgt_by)
    venstre, hoyre, bunn, topp = grenser

    ax_map.imshow(
        bilde,
        extent=(venstre, hoyre, bunn, topp),
        origin="upper",
        aspect="equal",
    )
    ax_map.set_xlim(venstre, hoyre)
    ax_map.set_ylim(bunn, topp)

    ax_map.set_xlabel("Lengdegrad")
    ax_map.set_ylabel("Breddegrad")
    ax_map.xaxis.set_major_formatter(FuncFormatter(lengdegrad))
    ax_map.yaxis.set_major_formatter(FuncFormatter(breddegrad))
    ax_map.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_map.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_map.tick_params(axis="x", labelrotation=25, pad=7, labelsize=9)
    ax_map.tick_params(axis="y", labelsize=9)
    ax_map.grid(
        True,
        linestyle=":",
        linewidth=0.7,
        color="#475467",
        alpha=0.55,
    )
    ax_map.set_title(
        f"Boligkart – {valgt_by}",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    for kant in ax_map.spines.values():
        kant.set_color("#667085")
        kant.set_linewidth(1.1)

    ax_history.clear()

    if marked_model_point is not None:
        model_x, model_y = marked_model_point

        # Convert the model's 0–100 point back to projected map coordinates.
        x = venstre + model_x / 100 * (hoyre - venstre)
        y = bunn + model_y / 100 * (topp - bunn)

        lon, lat = TIL_WGS84.transform(x, y)

        ax_map.scatter(
            x,
            y,
            s=150,
            color="#E63946",
            edgecolors="white",
            linewidths=2,
            zorder=5,
        )
        ax_map.annotate(
            f"Valgt posisjon\nBreddegrad: {lat:.5f}\nLengdegrad: {lon:.5f}",
            xy=(x, y),
            xytext=(14, 18),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.55",
                fc="white",
                ec="#E63946",
                alpha=0.96,
            ),
            arrowprops=dict(arrowstyle="->", color="#E63946"),
        )

        # Use the selected model for the current city.
        model = city_models[valgt_by][valgt_modell]
        m2list = [50, 100, 150, 200]

        price_per_m2 = (
            model.predict_price(
                model_x,
                model_y,
                m2,
                house_type,
                standard_type,
            )
            / m2
        )
        price_label = f"{round(price_per_m2 * 1000, -2):,.0f}"

        prices_per_m2 = [
            model.predict_price(
                model_x,
                model_y,
                size,
                house_type,
                standard_type,
            )
            / size
            for size in m2list
        ]

        ax_map.annotate(
            price_label,
            xy=(x, y),
            xytext=(8, -12),
            textcoords="offset points",
            color="blue",
            fontsize=12,
            zorder=6,
        )

        colors = {
            1: "red",
            2: "yellow",
            3: "green",
            4: "darkgreen",
        }

        ax_history.bar(
            m2list,
            prices_per_m2,
            color=colors.get(standard_type, "blue"),
            width=18,
        )
        ax_history.axhline(
            y=prices_per_m2[1],
            color="black",
            linestyle="--",
            label="Kvadratmeterpris ved 100 m²",
        )
        ax_history.grid(axis="y", linestyle=":", alpha=0.5)
        ax_history.legend(loc="upper right", fontsize=8)
    else:
        ax_history.text(
            0.5,
            0.5,
            "Klikk på kartet for å beregne pris",
            ha="center",
            va="center",
            transform=ax_history.transAxes,
        )

    ax_history.set_ylabel("Estimert pris per m² (1000 kr)")
    ax_history.set_xlabel("Boligstørrelse (m²)")
    ax_history.set_xticks([50, 100, 150, 200])

    fig.canvas.draw_idle()


def on_map_type(key):
    global valgt_by, marked_model_point
    valgt_by = key
    marked_model_point = None
    draw_the_map()


def on_model_type(key):
    global valgt_modell
    valgt_modell = key
    draw_the_map()


def on_house_type(key):
    global house_type
    house_type = house_types[key]
    draw_the_map()


def on_standard(key):
    global standard_type
    standard_type = standard_types[key]
    draw_the_map()


def on_click_map(event):
    global marked_model_point

    if (
        event.inaxes != ax_map
        or event.xdata is None
        or event.ydata is None
    ):
        return

    _, grenser = kartdata(valgt_by)
    marked_model_point = modellkoordinater(
        event.xdata,
        event.ydata,
        grenser,
    )

    lon, lat = TIL_WGS84.transform(event.xdata, event.ydata)
    print(f"{valgt_by}: breddegrad {lat:.6f}, lengdegrad {lon:.6f}")

    draw_the_map()


# Figure and axes
fig = plt.figure(figsize=(14, 10), facecolor="#F4F6F8")

ax_map = fig.add_axes([0.37, 0.07, 0.35, 0.84])
ax_history = fig.add_axes([0.07, 0.07, 0.23, 0.5])

ax_radio_map = fig.add_axes(
    [0.77, 0.84, 0.20, 0.10],
    facecolor="white",
)
ax_radio_house_type = fig.add_axes(
    [0.77, 0.64, 0.20, 0.16],
    facecolor="white",
)
ax_radio_standard = fig.add_axes(
    [0.77, 0.41, 0.20, 0.19],
    facecolor="white",
)
ax_radio_model = fig.add_axes(
    [0.77, 0.19, 0.20, 0.18],
    facecolor="white",
)

# Radio-button controls
radio_button_map = RadioButtons(ax_radio_map, list(BYER.keys()))
radio_button_house_type = RadioButtons(
    ax_radio_house_type,
    list(house_types.keys()),
)
radio_button_standard = RadioButtons(
    ax_radio_standard,
    list(standard_types.keys()),
)
radio_button_model = RadioButtons(
    ax_radio_model,
    model_names,
)

radio_button_map.on_clicked(on_map_type)
radio_button_house_type.on_clicked(on_house_type)
radio_button_standard.on_clicked(on_standard)
radio_button_model.on_clicked(on_model_type)

fig.canvas.mpl_connect("button_press_event", on_click_map)

draw_the_map()
plt.show()
