MAX_LIMIT = 9999


class StyleEmbedding:
    colorscheme = "tableau20"
    filled = True
    sizeDefault = 7
    sizeMin = 5
    sizeMax = 25
    opacityDefault = 0.05
    opacityMin = 0.05
    opacityMax = 0.5
    tickMinStep = 5

    def get_embedding_style():
        return {
            "config": {
                "axis": {
                    "labelFontSize": 20,
                    "labelSeparation": 10,
                    "titleFontSize": 24,
                },
                "mark": {"smooth": True},
                "legend": {
                    "titleFontSize": 24,
                    "labelFontSize": 20,
                    "titleLimit": MAX_LIMIT,
                    "labelLimit": MAX_LIMIT,
                    "symbolLimit": MAX_LIMIT,
                    "orient": "right",
                    # "orient": "top",
                    # "columns": 3,
                    # "direction": "horizontal",
                    "titleAnchor": "middle",
                    "labelOpacity": 1,
                    "symbolOpacity": 1,
                },
                "title": {"anchor": "start", "color": "gray", "fontSize": 25},
            }
        }


class StyleEthogram:
    colorscheme = "tableau20"

    def get_ethogram_style():
        return {
            "config": {
                "view": {"continuousWidth": 400, "continuousHeight": 300},
                "axis": {
                    "labelFontSize": 20,
                    "labelSeparation": 10,
                    "titleFontSize": 24,
                },
                "legend": {
                    "labelFontSize": 20,
                    "labelLimit": MAX_LIMIT,
                    "labelOpacity": 1,
                    "orient": "right",
                    "symbolLimit": MAX_LIMIT,
                    "symbolOpacity": 1,
                    "titleFontSize": 24,
                    "titleLimit": MAX_LIMIT,
                },
                "title": {"anchor": "start", "color": "gray", "fontSize": 25},
            }
        }
